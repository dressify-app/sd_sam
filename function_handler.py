import os
import io
import uuid
import base64
import time
import requests
import runpod
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
os.environ["ULTRALYTICS_SKIP_VALIDATE"] = "1"

try:
    from fastapi import FastAPI, Body
    import uvicorn
except ImportError:
    FastAPI = None
    uvicorn = None

# ──────────────────────────────────────────────────────────────────────
#  MediaPipe Models
# ──────────────────────────────────────────────────────────────────────
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_seg = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ──────────────────────────────────────────────────────────────────────
#  helpers: upload to S3
# ──────────────────────────────────────────────────────────────────────
def _upload_to_s3(image_data: bytes | str, *, source_type: str = "base64") -> str:
    import boto3
    s3_access_key = os.getenv("S3_ACCESS_KEY")
    s3_secret_key = os.getenv("S3_SECRET_KEY")
    s3_endpoint   = os.getenv("S3_ENDPOINT_URL")
    s3_bucket     = os.getenv("S3_BUCKET_NAME")
    s3_region_name = os.getenv("S3_REGION_NAME")
    if not all([s3_access_key, s3_secret_key, s3_endpoint, s3_bucket, s3_region_name]):
        raise ValueError("Missing S3 env vars")

    if source_type == "base64":
        if isinstance(image_data, str):
            if image_data.startswith("data:image"):
                image_data = image_data.split(",", 1)[1]
            img_bytes = base64.b64decode(image_data)
        else:
            raise ValueError("base64 must be str")
    else:  # source_type == "bytes"
        if isinstance(image_data, bytes):
            img_bytes = image_data
        else:
            raise ValueError("bytes source_type requires bytes input")

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=95)
    img_bytes = buf.getvalue()

    key = f"photo_out/generated/{uuid.uuid4().hex}/{int(time.time())}.jpg"
    session = boto3.session.Session()
    s3_client = session.client(
        's3',
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        endpoint_url=s3_endpoint,
        region_name=s3_region_name,
        config=boto3.session.Config(signature_version="s3"),
    )
    # Загрузка маски
    s3_client.put_object(
        Body=img_bytes,
        Bucket=s3_bucket,
        Key=key,
        ACL='public-read',
    )
    return f"{s3_endpoint.rstrip('/')}/{s3_bucket}/{key}"

# ──────────────────────────────────────────────────────────────────────
#  utils: base64 ⇄ cv2
# ──────────────────────────────────────────────────────────────────────
def _b64_to_cv2(b64: str) -> np.ndarray:
    if b64.startswith("data:image"):
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

def _cv2_to_png_bytes(mask: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", mask)
    return buf.tobytes()

def _get_pose_keypoints(img_b64: str, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Получает ключевые точки позы используя ControlNet OpenPose."""
    try:
        response = requests.post(
            "http://127.0.0.1:7860/controlnet/detect",
            json={
                "controlnet_module": "openpose",
                "controlnet_input_images": [img_b64]
            },
            timeout=30  # увеличиваем таймаут
        )
        response.raise_for_status()  # проверяем HTTP ошибки
        
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError(f"Invalid response format: {result}")
            
        if "images" not in result:
            raise ValueError(f"No pose maps in response. Full response: {result}")
            
        if not result["images"] or not isinstance(result["images"], list):
            raise ValueError(f"Empty pose maps in response: {result}")
            
        # Получаем карту позы
        pose_map = _b64_to_cv2(result["images"][0])
        if pose_map is None or pose_map.size == 0:
            raise ValueError("Failed to decode pose map image")
        
        # Конвертируем в градации серого для поиска ключевых точек
        gray = cv2.cvtColor(pose_map, cv2.COLOR_BGR2GRAY)
        
        # Находим контуры
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No contours found in pose map")
        
        # Берем самый большой контур
        main_contour = max(contours, key=cv2.contourArea)
        
        # Получаем ограничивающий прямоугольник
        # Размер карты, на которой работал ControlNet
        if "poses" in result and result["poses"]:
            canvas_w = result["poses"][0].get("canvas_width", pose_map.shape[1])
            canvas_h = result["poses"][0].get("canvas_height", pose_map.shape[0])
        else:
            canvas_w, canvas_h = pose_map.shape[1], pose_map.shape[0]

        # Масштабы в размер оригинального изображения
        orig_h, orig_w = img_bgr.shape[:2]          # <-- передайте сюда img_bgr
        sx, sy = orig_w / canvas_w, orig_h / canvas_h

        # bounding box из контура
        x, y, w, h = cv2.boundingRect(main_contour)
        box_xyxy = np.array([x * sx, y * sy, (x + w) * sx, (y + h) * sy]).astype(int)
        
        # Сначала пробуем взять готовые keypoints из JSON
        if "poses" in result and result["poses"]:
            kp_raw = result["poses"][0]["people"][0]["pose_keypoints_2d"]
            keypoints = np.array(kp_raw, dtype=np.float32).reshape(-1, 3)[:, :2]
            keypoints[:, 0] *= sx
            keypoints[:, 1] *= sy
        else:
            keypoints = []
            for _ in range(17):
                keypoints.append(0)
            keypoints = np.array(keypoints, dtype=np.float32)

        return keypoints.astype(int), box_xyxy
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"ControlNet request failed: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to process pose detection: {str(e)}")

# ──────────────────────────────────────────────────────────────────────
#  generate_body_mask: MediaPipe segmentation + FaceMesh head removal
# ──────────────────────────────────────────────────────────────────────
def generate_body_mask(img_b64: str, dilate_size: int = 15) -> tuple[str, dict]:
    img_bgr = _b64_to_cv2(img_b64)
    h, w = img_bgr.shape[:2]

    # Full-body mask via SelfieSegmentation
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    seg_res = mp_seg.process(rgb)
    mask_body = (seg_res.segmentation_mask > 0.5).astype(np.uint8)

    # Head mask via FaceMesh (circle around forehead)
    mask_head = np.zeros_like(mask_body, dtype=np.uint8)
    face_res = mp_face.process(rgb)
    if face_res.multi_face_landmarks:
        lm = face_res.multi_face_landmarks[0].landmark
        x_f, y_f = int(lm[10].x * w), int(lm[10].y * h)
        x_c, y_c = int(lm[152].x * w), int(lm[152].y * h)
        r = int(np.hypot(x_c - x_f, y_c - y_f) * 1.1)
        cv2.circle(mask_head, (x_f, y_f), r, 1, -1)

    # Subtract head
    mask_final = cv2.bitwise_and(mask_body, cv2.bitwise_not(mask_head))

    # Optional dilation
    if dilate_size and dilate_size > 0:
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        mask_final = cv2.dilate(mask_final, kernel, iterations=1)

    # Upload
    png = _cv2_to_png_bytes((mask_final * 255).astype(np.uint8))
    url = _upload_to_s3(png, source_type='bytes')
    debug = {
        "dilate_size": dilate_size,
        "face_detected": bool(face_res.multi_face_landmarks)
    }
    return url, debug

# ──────────────────────────────────────────────────────────────────────
#  Proxy / request router
# ──────────────────────────────────────────────────────────────────────
def _maybe_fetch(val):
    if isinstance(val, str) and val.startswith(("http://", "https://")):
        resp = requests.get(val, timeout=10)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode()
    return val

def _process_sd_results(response_data):
    """
    Обрабатывает результаты от SD API, загружает изображения в S3 и возвращает обновленный ответ.
    """
    try:
        imgs = None
        if 'output' in response_data and isinstance(response_data['output'], dict):
            out = response_data['output']
            if 'images' in out and isinstance(out['images'], list):
                imgs = out['images']
        if not imgs and 'images' in response_data and isinstance(response_data['images'], list):
            imgs = response_data['images']
        if imgs:
            uploaded = []
            for data in imgs:
                # data may be base64
                img_b64 = data if data.startswith("data:image") else f"data:image/png;base64,{data}"
                cv2_img = _b64_to_cv2(img_b64)
                _, buf = cv2.imencode(".jpg", cv2_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                uploaded.append(_upload_to_s3(buf.tobytes(), source_type='bytes'))
            if 'output' in response_data:
                response_data['output']['images'] = uploaded
            else:
                response_data['images'] = uploaded
            response_data['result_url'] = uploaded[0]
        return response_data
    except Exception:
        return response_data

def _process_sam_results(response_data):
    # Similar to SD results, but for SAM-style masks lists
    try:
        out = response_data.get('output', {})
        for key in ('masks','masked_images','blended_images'):
            if key in out and isinstance(out[key], list):
                uploaded = []
                for data in out[key]:
                    img_b64 = data if data.startswith("data:image") else f"data:image/png;base64,{data}"
                    cv2_img = _b64_to_cv2(img_b64)
                    _, buf = cv2.imencode(".png", cv2_img)
                    uploaded.append(_upload_to_s3(buf.tobytes(), source_type='bytes'))
                out[key] = uploaded
                response_data['result_url'] = uploaded[-1]
        return response_data
    except Exception:
        return response_data

def process_request(job: dict):
    inp = job.get("input", {})
    path = inp.get("path", "")
    params = inp.get("params", {})

    if path == "fast-mask/body":
        img_src = params.get("input_image")
        if not img_src:
            return {"error": "'input_image' is required"}
        # convert URL→base64 if needed
        img_src = _maybe_fetch(img_src)
        try:
            dil = int(params.get("dilate_size", 15))
            url, dbg = generate_body_mask(img_src, dil)
            return {"result_url": url, "debug": dbg}
        except Exception as e:
            return {"error": str(e)}

    # proxy all other requests to local AUTOMATIC1111 WebUI
    # fetch any URL fields
    if not path:
        return {"error": "Missing 'path' in input"}
    for k, v in list(params.items()):
        lk = k.lower()
        if (lk.endswith("image") or lk == "mask") and isinstance(v, str):
            params[k] = _maybe_fetch(v)
        elif lk.endswith("images") and isinstance(v, list):
            params[k] = [_maybe_fetch(x) for x in v]

    if path in ["sdapi/v1/txt2img", "sdapi/v1/img2img"]:
        # Получаем исходное изображение
        input_image = params.get("init_images", [None])[0] if path == "sdapi/v1/img2img" else None
        
        if input_image:
            # Создаем ControlNet units для глубины и позы
            controlnet_units = []
            
            # Depth ControlNet
            depth_response = requests.post(
                "http://127.0.0.1:7860/controlnet/detect",
                json={
                    "controlnet_module": "depth",
                    "controlnet_input_images": [input_image]
                }
            )
            if depth_response.ok:
                depth_result = depth_response.json()
                if "depth_maps" in depth_result:
                    controlnet_units.append({
                        "input_image": depth_result["depth_maps"][0],
                        "module": "depth",
                        "model": "control_v11f1p_sd15_depth",
                        "weight": 0.6,  # Уменьшаем вес для меньшего влияния на цвет
                        "resize_mode": "Resize and Fill",
                        "lowvram": False,
                        "processor_res": 512,
                        "threshold_a": 64,
                        "threshold_b": 64,
                        "guidance_start": 0.0,
                        "guidance_end": 1.0,
                        "control_mode": "Balanced"
                    })
            
            # Pose ControlNet
            pose_response = requests.post(
                "http://127.0.0.1:7860/controlnet/detect",
                json={
                    "controlnet_module": "openpose",
                    "controlnet_input_images": [input_image]
                }
            )
            if pose_response.ok:
                pose_result = pose_response.json()
                if "images" in pose_result:
                    # Получаем ключевые точки
                    try:
                        keypoints, box_xyxy = _get_pose_keypoints(input_image, _b64_to_cv2(input_image))
                        # Добавляем ключевые точки в параметры ControlNet
                        controlnet_units.append({
                            "input_image": pose_result["images"][0],
                            "module": "openpose",
                            "model": "control_v11p_sd15_pose",
                            "weight": 0.8,
                            "resize_mode": "Resize and Fill",
                            "lowvram": False,
                            "processor_res": 512,
                            "threshold_a": 64,
                            "threshold_b": 64,
                            "guidance_start": 0.0,
                            "guidance_end": 1.0,
                            "control_mode": "Balanced",
                            "keypoints": keypoints.tolist(),
                            "bounding_box": box_xyxy.tolist()
                        })
                    except Exception as e:
                        print(f"Warning: Failed to get pose keypoints: {str(e)}")
                        # Если не удалось получить ключевые точки, используем стандартный ControlNet
                        controlnet_units.append({
                            "input_image": pose_result["images"][0],
                            "module": "openpose",
                            "model": "control_v11p_sd15_pose",
                            "weight": 0.8,
                            "resize_mode": "Resize and Fill",
                            "lowvram": False,
                            "processor_res": 512,
                            "threshold_a": 64,
                            "threshold_b": 64,
                            "guidance_start": 0.0,
                            "guidance_end": 1.0,
                            "control_mode": "Balanced"
                        })
            
            # Добавляем ControlNet units в параметры
            if controlnet_units:
                params["alwayson_scripts"] = {
                    "controlnet": {
                        "args": controlnet_units
                    }
                }

            # Добавляем параметры для лучшего сохранения деталей
            if "override_settings" not in params:
                params["override_settings"] = {}
            params["override_settings"].update({
                "CLIP_stop_at_last_layers": 2,  # Уменьшаем влияние CLIP на цвет
                "img2img_fix_steps": True,  # Фиксируем шаги для лучшей стабильности
                "img2img_background_color": "white"  # Устанавливаем белый фон
            })

            # Добавляем параметры для лучшего сохранения деталей
            if "script_args" not in params:
                params["script_args"] = []
            params["script_args"].extend([
                "Detailer",  # Включаем Detailer для лучшего сохранения деталей
                {
                    "detailer_denoising_strength": 0.3,  # Уменьшаем силу денойзинга
                    "detailer_prompt": "skin, face, hands, detailed skin texture",  # Фокус на коже
                    "detailer_negative_prompt": "deformed skin, bad skin, wrong skin color"  # Избегаем проблем с кожей
                }
            ])

    local_url = f"http://127.0.0.1:7860/{path.lstrip('/')}"
    try:
        resp = requests.post(local_url, json=params, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        if 'sdapi/v1' in path:
            result = _process_sd_results(result)
        elif 'sam' in path:
            result = _process_sam_results(result)
        elif 'controlnet' in path:
            # Обработка результатов ControlNet API
            result = _process_sd_results(result)
        
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if os.getenv("LOCAL") and FastAPI and uvicorn:
        app = FastAPI()

        @app.post("/run")
        def run_job(job: dict = Body(...)):  # type: ignore[valid-type]
            if "path" in job and "params" in job:
                job = {"input": job}
            return process_request(job)

        uvicorn.run(app, host="0.0.0.0", port=8080)
    else:
        runpod.serverless.start({"handler": process_request})