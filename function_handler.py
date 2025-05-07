import runpod
import os
import requests
import base64
import time
import uuid
import io
from PIL import Image
import numpy as np
import torch
import cv2
from ultralytics import YOLO               # yolov8‑pose
from segment_anything import (             # стандартный SAM
    sam_model_registry,
    SamAutomaticMaskGenerator,
)

os.environ["ULTRALYTICS_SKIP_VALIDATE"] = "1"

try:
    from fastapi import FastAPI, Body
    import uvicorn
except ImportError:                         # локальный режим может быть без этих пакетов
    FastAPI = None                          # type: ignore
    uvicorn = None                          # type: ignore

# ============================================================
# helpers: S3 upload + base64 utilities
# ============================================================

def _upload_to_s3(image_data: bytes | str, *, source_type: str = "base64") -> str:
    """Upload image/mask bytes to S3 and return public URL."""
    import boto3

    s3_access_key = os.getenv("S3_ACCESS_KEY")
    s3_secret_key = os.getenv("S3_SECRET_KEY")
    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")
    s3_region_name = os.getenv("S3_REGION_NAME", "us-east-1")
    s3_bucket_name = os.getenv("S3_BUCKET_NAME")
    if not all([s3_access_key, s3_secret_key, s3_endpoint_url, s3_bucket_name]):
        raise ValueError("Missing S3 env vars")

    # decode / normalise
    if source_type == "base64":
        if isinstance(image_data, str):
            if image_data.startswith("data:image"):
                image_data = image_data.split(",", 1)[1]
            img_bytes = base64.b64decode(image_data)
        else:
            raise ValueError("Base64 image data must be str")
    elif source_type == "bytes":
        img_bytes = image_data  # type: ignore[arg-type]
    else:
        raise ValueError("source_type must be 'base64' or 'bytes'")

    # force JPEG for size
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    img_bytes = buf.getvalue()

    ts = int(time.time())
    fid = uuid.uuid4().hex
    key_path = f"photo_out/generated/{fid}/{ts}.jpg"

    session = boto3.session.Session()
    s3 = session.client(
        "s3",
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        endpoint_url=s3_endpoint_url,
        region_name=s3_region_name,
        config=boto3.session.Config(signature_version="s3"),
    )
    s3.put_object(
        Body=img_bytes,
        Bucket=s3_bucket_name,
        Key=key_path,
        ACL="public-read",
        ContentType="image/jpeg",
    )
    return f"{s3_endpoint_url.rstrip('/')}/{s3_bucket_name}/{key_path}"

# ============================================================
# Fast body‑mask pipeline: YOLOv8‑pose + Segment‑Anything
# ============================================================

device = "cuda"  # или "cpu"

yolo_pose = YOLO("yolov8x-pose.pt")  # <- загрузка уже включает веса
yolo_pose.model.to(device).eval()    # переносим **внутреннюю** модель и ставим eval
try:
    yolo_pose.fuse()                 # ~10 % speed‑up
except Exception:
    pass

# --- SAM ----------------------------------------------------------------
sam_ckpt = os.getenv("SAM_CKPT", "sam_vit_b_01ec64.pth")  # vit‑b по‑умолчанию
sam_model = sam_model_registry["vit_b"](checkpoint=sam_ckpt).to(device).eval()
mask_gen = SamAutomaticMaskGenerator(
    sam_model,
    points_per_side=16,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
)

def _b64_to_cv2(img_b64: str) -> np.ndarray:
    if img_b64.startswith("data:image"):
        img_b64 = img_b64.split(",", 1)[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)


def generate_body_mask(img_b64: str) -> tuple[str, dict]:
    """Generates mask of entire body except head, hair & shoes; returns (S3 url, debug)."""
    img_bgr = _b64_to_cv2(img_b64)
    h, w = img_bgr.shape[:2]

    # 1. Detect person + keypoints -------------------------------------------------
    pose = yolo_pose.predict(img_bgr, conf=0.25, iou=0.5, verbose=False, device=device)[0]
    if len(pose.boxes) == 0:
        raise ValueError("No person detected")
    idx = int(torch.argmax(pose.boxes.conf))
    box_xyxy = pose.boxes.xyxy[idx].cpu().numpy().astype(int)
    keypts = pose.keypoints.xy[idx].cpu().numpy().astype(int)  # (17, 2)

    x1, y1, x2, y2 = box_xyxy
    crop = img_bgr[y1:y2, x1:x2]

    # 2. SAM full‑body mask -------------------------------------------------
    masks = mask_gen.generate(crop)

    crop_h, crop_w = crop.shape[:2]
    crop_area = crop_h * crop_w

    # --- helper: проверяем, попадает ли хотя бы 4 ключ‑точки в маску -------
    def _kp_covered(seg: np.ndarray, pts: np.ndarray, thresh: int = 4) -> bool:
        ok = 0
        for x, y in pts:
            if x <= 0 or y <= 0:
                continue
            if y - y1 >= 0 and x - x1 >= 0 and y - y1 < seg.shape[0] and x - x1 < seg.shape[1]:
                ok += bool(seg[int(y - y1), int(x - x1)])
        return ok >= thresh

    # 2a. собираем кандидатов: не > 80 % кропа и закрывают ≥4 ключ‑точки
    candidates: list[np.ndarray] = []
    for m in masks:
        seg = m["segmentation"]
        if m["area"] > crop_area * 0.80:      # почти весь кроп = фон → пропускаем
            continue
        if _kp_covered(seg, keypts):
            candidates.append(seg)

    # 2b. если нашлось ≥1 кандидата → объединяем; иначе fallback: берём 2‑ю по площади
    if candidates:
        body_mask = np.zeros_like(candidates[0], dtype=bool)
        for seg in candidates:
            body_mask |= seg
    else:
        masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)
        body_mask = masks_sorted[1]["segmentation"] if len(masks_sorted) > 1 else masks_sorted[0]["segmentation"]
    body_mask = body_mask.astype(bool)

    # ──────────────────────────────────────────────────────────────────────
    # 3.  Строим маску "запретных" зон: head+hair и shoes
    # ──────────────────────────────────────────────────────────────────────

    # 3a.  SHOES  – как было
    ank_idx = [15, 16]
    ank_pts = keypts[ank_idx]
    valid_ank = (ank_pts[:, 1] > 0)
    shoes_mask = np.zeros_like(body_mask)
    if valid_ank.any():
        ay = int(ank_pts[valid_ank, 1].max())
        sy1 = max(ay - y1, 0)
        shoes_mask[sy1:, :] = True

    # 3b.  HEAD + HAIR  через линию плеч
    shldr_idx  = [5, 6]                     # COCO: L/R shoulder
    shldr_pts  = keypts[shldr_idx]
    valid_shld = (shldr_pts[:, 1] > 0) & (shldr_pts[:, 0] > 0)

    head_mask = np.zeros_like(body_mask)

    if valid_shld.any():
        # средняя высота плеч относительно кропа
        y_shldr = int(shldr_pts[valid_shld, 1].mean()) - y1

        # запас: +8 % высоты туловища (для воротников, шарфов)
        margin  = int(0.08 * body_mask.shape[0])
        cut_y   = max(y_shldr - margin, 0)

        head_mask[:cut_y, :] = True                # всё выше линии плеч → True
    else:
        # резервный вариант, если плеч не нашли: нос/глаза (как раньше)
        head_idx = [0, 1, 2, 3, 4]
        head_pts = keypts[head_idx]
        valid_head = (head_pts[:, 0] > 0) & (head_pts[:, 1] > 0)
        if valid_head.any():
            hp = head_pts[valid_head]
            hx1, hy1 = hp.min(axis=0);  hx2, hy2 = hp.max(axis=0)
            head_h   = max(hy2 - hy1, 1)
            up       = int(head_h * 1.5)
            side     = int(head_h * 0.3)

            hx1 = max(hx1 - side - x1, 0)
            hx2 = min(hx2 - x1 + side, crop.shape[1] - 1)
            hy1 = max(hy1 - up   - y1, 0)
            hy2 = min(hy2 - y1 + int(head_h * 0.3), crop.shape[0] - 1)
            head_mask[hy1:hy2, hx1:hx2] = True

    # ──────────────────────────────────────────────────────────────────────
    # 4.  Dilate -> Subtract -> Final mask
    # ──────────────────────────────────────────────────────────────────────
    body_dil = cv2.dilate(body_mask.astype(np.uint8),
                          np.ones((11, 11), np.uint8), 2).astype(bool)

    final_crop = np.logical_and(body_dil,
                 np.logical_not(np.logical_or(head_mask, shoes_mask)))

    # ──────────────────────────────────────────────────────────────────────
    # 5.  Вставляем в канву и отправляем в S3 (осталось как было)
    # ──────────────────────────────────────────────────────────────────────
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = final_crop.astype(np.uint8) * 255

    _, png_bytes = cv2.imencode(".png", full_mask)
    mask_url = _upload_to_s3(png_bytes.tobytes(), source_type="bytes")

    debug = {
        "bbox": box_xyxy.tolist(),
        "shoulders_used": bool(valid_shld.any()),
        "ankles_used":    bool(valid_ank.any()),
    }
    return mask_url, debug

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