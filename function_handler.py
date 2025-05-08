import runpod
import os, requests, base64, time, uuid, io
from PIL import Image
import numpy as np
import torch, cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# ──────────────────────────────────────────────────────────────────────
#  Models
# ──────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

sam_ckpt = os.getenv("SAM_CKPT", "sam_vit_b_01ec64.pth")
sam_model = sam_model_registry["vit_b"](checkpoint=sam_ckpt).to(device).eval()
predictor = SamPredictor(sam_model)
os.environ["ULTRALYTICS_SKIP_VALIDATE"] = "1"

try:
    from fastapi import FastAPI, Body
    import uvicorn
except ImportError:
    FastAPI = None
    uvicorn = None

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
    else:
        img_bytes = image_data               # bytes

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
        ContentType='image/jpeg'
    )
    return f"{s3_endpoint.rstrip('/')}/{s3_bucket}/{key}"   

# ──────────────────────────────────────────────────────────────────────
#  utils
# ──────────────────────────────────────────────────────────────────────
def _b64_to_cv2(b64: str) -> np.ndarray:
    if b64.startswith("data:image"):
        b64 = b64.split(",", 1)[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8),
                        cv2.IMREAD_COLOR)

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
            # fallback – максимумы по PNG (как было)
            keypoints = []
            for _ in range(17):
                ...
            keypoints = np.array(keypoints, dtype=np.float32)

        return keypoints.astype(int), box_xyxy
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"ControlNet request failed: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to process pose detection: {str(e)}")

# ──────────────────────────────────────────────────────────────────────
#  main mask generator
# ──────────────────────────────────────────────────────────────────────
def generate_body_mask(img_b64: str, dilate_size: int = 0) -> tuple[str, dict]:
    img_bgr = _b64_to_cv2(img_b64)
    h, w = img_bgr.shape[:2]

    # 1) детектим keypoints + bbox как раньше
    keypts, box_xyxy = _get_pose_keypoints(img_b64, img_bgr)
    x1, y1, x2, y2 = box_xyxy
    crop = img_bgr[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]

    # 2) передаём crop в predictor
    predictor.set_image(crop)

    # box в координатах crop: весь прямоугольник от (0,0) до (crop_w, crop_h)
    input_box = np.array([[0, 0, crop_w, crop_h]])
    masks, scores, logits = predictor.predict(
        box=input_box,
        multimask_output=False
    )
    # единственная маска тела
    body_mask = masks[0].astype(bool)

    # 3) убираем голову+волосы: рисуем круг/овал по keypoints
    # создаём маску головы в crop-координатах
    head_pts = keypts[[0,1,2,3,4]] - np.array([x1, y1])  # переносим в координаты crop
    valid = (head_pts[:,0]>0)&(head_pts[:,1]>0)
    head_mask = np.zeros_like(body_mask, dtype=bool)
    if valid.any():
        hp = head_pts[valid]
        cx, cy = hp[:,0].mean(), hp[:,1].mean()
        radius = max(hp[:,0].ptp(), hp[:,1].ptp()) * 0.6
        cv2.circle(head_mask, (int(cx),int(cy)), int(radius), 1, -1)

    # итоговая маска корпуса без головы
    final_crop = body_mask & ~head_mask

    # 4) опционально дилатация
    if dilate_size>0:
        final_crop = cv2.dilate(final_crop.astype(np.uint8),
                                np.ones((dilate_size, dilate_size), np.uint8),
                                iterations=1).astype(bool)

    # 5) кладём обратно на full-size
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = final_crop.astype(np.uint8) * 255

    # 6) загрузка и отладка
    _, png = cv2.imencode('.png', full_mask)
    url = _upload_to_s3(png.tobytes(), source_type='bytes')
    dbg = {'bbox': box_xyxy.tolist(), 'head_cut': True}
    return url, dbg
# ============================================================
# Proxy / request router
# ============================================================

def _process_sd_results(response_data):
    """
    Обрабатывает результаты от SD API, загружает изображения в S3 и возвращает обновленный ответ.
    """
    try:
        # Проверяем разные форматы ответа
        images = None
        
        # Проверяем вложенную структуру в output.images (основной формат)
        if 'output' in response_data and isinstance(response_data['output'], dict):
            output = response_data['output']
            if 'images' in output and isinstance(output['images'], list):
                images = output['images']
        
        # Проверяем корневое поле images (альтернативный формат)
        if not images and 'images' in response_data and isinstance(response_data['images'], list):
            images = response_data['images']
            
        # Если нашли изображения, обрабатываем их
        if images and len(images) > 0:
            uploaded_images = []
            for i, img_data in enumerate(images):
                url = _upload_to_s3(img_data)
                uploaded_images.append(url)
                
            # Добавляем URL в оба формата для совместимости
            if 'output' in response_data and isinstance(response_data['output'], dict):
                response_data['output']['images'] = uploaded_images
            else:
                response_data['images'] = uploaded_images
                
            # Добавляем поле с первым URL для легкого доступа
            response_data['result_url'] = uploaded_images[0] if uploaded_images else None
            
        return response_data
    except Exception as e:
        return {"error": f"Failed to process SD results: {e}", "original_response": response_data}


def _process_sam_results(response_data):
    """
    Обрабатывает результаты от SAM API, загружает маски в S3 и возвращает обновленный ответ.
    """
    try:
        # Проверяем вложенную структуру в output
        if 'output' in response_data and isinstance(response_data['output'], dict):
            output = response_data['output']
            # Проходим по всем возможным ключам с масками
            for key in ['masks', 'masked_images', 'blended_images']:
                if key in output and output[key]:
                    masks = output[key]
                    if isinstance(masks, list) and masks:
                        uploaded_masks = []
                        for i, mask_data in enumerate(masks):
                            url = _upload_to_s3(mask_data)
                            uploaded_masks.append(url)
                        # Обновляем маски в output
                        output[key] = uploaded_masks
                        # Добавляем поле result_url для легкого доступа
                        if 'result_url' not in response_data:
                            response_data['result_url'] = uploaded_masks[-1] if uploaded_masks else None
        
        # Для обратной совместимости проверяем также корневые поля
        for key in ['masks', 'masked_images', 'blended_images']:
            if key in response_data and response_data[key]:
                masks = response_data[key]
                if isinstance(masks, list) and masks:
                    uploaded_masks = []
                    for i, mask_data in enumerate(masks):
                        url = _upload_to_s3(mask_data)
                        uploaded_masks.append(url)
                    # Обновляем маски в корневом поле
                    response_data[key] = uploaded_masks
                    # Добавляем поле result_url для легкого доступа если его еще нет
                    if 'result_url' not in response_data:
                        response_data['result_url'] = uploaded_masks[-1] if uploaded_masks else None
        
        return response_data
    except Exception as e:
        return {"error": f"Failed to process SAM results: {e}", "original_response": response_data}

def process_request(job: dict):
    input_data = job.get("input", {})
    path = input_data.get("path", "")
    params = input_data.get("params", {})

    # Fast mask endpoint
    if path == "fast-mask/body":
        img_src = params.get("input_image")
        if not img_src:
            return {"error": "'input_image' is required"}
        if img_src.startswith("http://") or img_src.startswith("https://"):
            r = requests.get(img_src, timeout=10)
            r.raise_for_status()
            img_b64 = base64.b64encode(r.content).decode()
        else:
            img_b64 = img_src
        try:
            dilate_size = int(params.get("dilate_size", 15))
            url, dbg = generate_body_mask(img_b64, dilate_size)
            return {"result_url": url, "debug": dbg}
        except Exception as e:
            return {"error": str(e)}

    # ---- other paths: proxy to local WebUI ----
    if not path:
        return {"error": "Missing 'path' in input"}
    # Автоматическая конвертация URL-изображений в base64 для всех полей image/images
    def _maybe_fetch(val):
        if isinstance(val, str) and val.startswith(("http://", "https://")):
            resp = requests.get(val)
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode()
        return val

    for key, val in list(params.items()):
        lk = key.lower()
        # одиночные поля, оканчивающиеся на image
        if (lk.endswith('image') or lk == 'mask') and isinstance(val, str):
            try:
                params[key] = _maybe_fetch(val)
            except Exception as e:
                return {"error": f"Failed to fetch '{key}': {e}"}
        # списки изображений, оканчивающиеся на images
        elif lk.endswith('images') and isinstance(val, list):
            new_list = []
            for item in val:
                try:
                    new_list.append(_maybe_fetch(item))
                except Exception as e:
                    return {"error": f"Failed to fetch element of '{key}': {e}"}
            params[key] = new_list

    # Добавляем ControlNet для txt2img и img2img
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

    # Build local URL
    local_url = f"http://127.0.0.1:7860/{path.lstrip('/')}"
    try:
        resp = requests.post(local_url, json=params)
        resp.raise_for_status()
        result = resp.json()
        
        # Обработка результатов - загрузка в S3 и получение URL
        if 'sdapi/v1' in path:
            # Обработка результатов SD API
            result = _process_sd_results(result)
        elif 'sam' in path:
            # Обработка результатов SAM API
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