import runpod
import os, requests, base64, time, uuid, io
from PIL import Image
import numpy as np
import torch, cv2
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

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
    s3 = boto3.client(
        "s3",
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        endpoint_url=s3_endpoint,
        region_name=s3_region_name,
        config=boto3.session.Config(signature_version="s3"),
    )
    s3.put_object(Body=img_bytes, Bucket=s3_bucket,
                  Key=key, ACL="public-read", ContentType="image/jpeg")
    return f"{s3_endpoint.rstrip('/')}/{s3_bucket}/{key}"

# ──────────────────────────────────────────────────────────────────────
#  Models
# ──────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_pose = YOLO("yolov8x-pose.pt")
yolo_pose.model.to(device).eval()
try: yolo_pose.fuse()
except: pass

sam_ckpt = os.getenv("SAM_CKPT", "sam_vit_b_01ec64.pth")
sam_model = sam_model_registry["vit_b"](checkpoint=sam_ckpt).to(device).eval()
mask_gen  = SamAutomaticMaskGenerator(
    sam_model,
    points_per_side=32,                 # чуть плотнее
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
)

# ──────────────────────────────────────────────────────────────────────
#  utils
# ──────────────────────────────────────────────────────────────────────
def _b64_to_cv2(b64: str) -> np.ndarray:
    if b64.startswith("data:image"):
        b64 = b64.split(",", 1)[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8),
                        cv2.IMREAD_COLOR)

# ──────────────────────────────────────────────────────────────────────
#  main mask generator
# ──────────────────────────────────────────────────────────────────────
def generate_body_mask(img_b64: str) -> tuple[str, dict]:
    """Generates mask of entire body except head, hair & shoes; returns (S3 url, debug)."""
    img_bgr = _b64_to_cv2(img_b64)
    h, w = img_bgr.shape[:2]

    # 1. Detect person + keypoints -------------------------------------------------
    pose = yolo_pose.predict(img_bgr, conf=0.25, iou=0.5, verbose=False, device=device)[0]
    if len(pose.boxes) == 0:
        raise ValueError("No person detected")
    idx       = int(torch.argmax(pose.boxes.conf))
    box_xyxy  = pose.boxes.xyxy[idx].cpu().numpy().astype(int)
    keypts    = pose.keypoints.xy[idx].cpu().numpy().astype(int)  # (17,2)

    x1, y1, x2, y2 = box_xyxy
    crop   = img_bgr[y1:y2, x1:x2]

    # 2. SAM full‑body mask -------------------------------------------------
    masks = mask_gen.generate(crop)
    crop_h, crop_w = crop.shape[:2]
    crop_area = crop_h * crop_w

    # --- helper: проверяем, попадает ли хотя бы 4 ключ‑точки в маску -------
    def _kp_covered(seg: np.ndarray, pts: np.ndarray, thresh=4) -> bool:
        ok = 0
        for x, y in pts:
            if x <= 0 or y <= 0: continue
            if 0 <= y - y1 < seg.shape[0] and 0 <= x - x1 < seg.shape[1]:
                ok += bool(seg[int(y - y1), int(x - x1)])
        return ok >= thresh

    candidates = []
    for m in masks:
        seg = m["segmentation"]
        if m["area"] > crop_area * 0.80:  # отбрасываем почти‑фон
            continue
        if _kp_covered(seg, keypts):
            candidates.append(seg)

    if candidates:
        body_mask = np.zeros_like(candidates[0], dtype=bool)
        for seg in candidates: body_mask |= seg
    else:
        m_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)
        body_mask = m_sorted[1]["segmentation"] if len(m_sorted) > 1 else m_sorted[0]["segmentation"]
    body_mask = body_mask.astype(bool)

    # ──────────────────────────────────────────────────────────────────────
    # 3.  Строим маску "запретных" зон: head+hair и shoes
    # ──────────────────────────────────────────────────────────────────────

    # 3a.  SHOES  – как было
    ank_idx = [15, 16]
    ank_pts = keypts[ank_idx]
    valid_ank = (ank_pts[:, 1] > 0)
    shoes_mask = np.zeros_like(body_mask, dtype=bool)
    if valid_ank.any():
        ay  = int(ank_pts[valid_ank, 1].max())
        sy1 = max(ay - y1, 0)
        shoes_mask[sy1:, :] = True

    # 4. head+hair  -------------------------------------------------------
    head_mask_uint = np.zeros(crop.shape[:2], dtype=np.uint8, order="C").copy()
    head_idx = [0, 1, 2, 3, 4]
    head_pts = keypts[head_idx]
    valid_head = (head_pts[:, 0] > 0) & (head_pts[:, 1] > 0)

    if valid_head.any():
        hp = head_pts[valid_head]
        xmin, ymin = hp.min(axis=0)
        xmax, ymax = hp.max(axis=0)
        cx = int((xmin + xmax) / 2) - x1
        cy = int((ymin + ymax) / 2) - y1
        head_h = ymax - ymin
        head_w = xmax - xmin
        radius = int(max(head_h, head_w) * 0.9)
        cy -= int(head_h * 0.25)
        # draw filled circle
        cv2.circle(head_mask_uint, (cx, cy), radius, 1, thickness=-1)
    else:
        # fallback: very local shoulder cut (only if absolutely needed)
        sh_idx = [5, 6]
        sh_pts = keypts[sh_idx]
        valid_sh = (sh_pts[:, 0] > 0) & (sh_pts[:, 1] > 0)
        if valid_sh.any():
            y_sh = int(sh_pts[valid_sh, 1].mean()) - y1
            margin = int(0.05 * body_mask.shape[0])
            head_mask_uint[:max(y_sh - margin, 0), :] = 1

    head_mask = head_mask_uint.astype(bool)
    # 5. final ------------------------------------------------------------
    body_dil = cv2.dilate(body_mask.astype(np.uint8),
                          np.ones((13,13),np.uint8), 2).astype(bool)
    final_crop = np.logical_and(body_dil,
                    np.logical_not(np.logical_or(head_mask, shoes_mask)))

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = final_crop.astype(np.uint8) * 255

    _, png_bytes = cv2.imencode(".png", full_mask)
    url = _upload_to_s3(png_bytes.tobytes(), source_type="bytes")

    dbg = {"bbox": box_xyxy.tolist(),
           "head_used": bool(valid_head.any()),
           "ankles_used": bool(valid_ank.any())}
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
            url, dbg = generate_body_mask(img_b64)
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