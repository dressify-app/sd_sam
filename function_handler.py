import runpod
import os
import requests
import base64
import time
import uuid
import io
from PIL import Image
try:
    from fastapi import FastAPI, Body
    import uvicorn
except ImportError:
    FastAPI = None
    uvicorn = None

def _upload_to_s3(image_data, source_type='base64'):
    """
    Загружает изображение в S3 и возвращает публичный URL.
    
    Args:
        image_data: данные изображения в формате base64 или bytes
        source_type: тип исходных данных - 'base64' или 'bytes'
    
    Returns:
        str: публичный URL загруженного изображения
    """
    import boto3
    
    # Получаем переменные окружения для S3
    s3_access_key = os.getenv('S3_ACCESS_KEY')
    s3_secret_key = os.getenv('S3_SECRET_KEY')
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    s3_region_name = os.getenv('S3_REGION_NAME', 'us-east-1')
    s3_bucket_name = os.getenv('S3_BUCKET_NAME')
    
    # Проверка наличия всех необходимых переменных
    if not all([s3_access_key, s3_secret_key, s3_endpoint_url, s3_bucket_name]):
        raise ValueError("Missing required S3 environment variables")
    
    # Подготовка данных изображения
    if source_type == 'base64':
        # Обработка base64 разных форматов (data URI или сырой base64)
        if isinstance(image_data, str):
            if image_data.startswith('data:image'):
                image_data = image_data.split(',', 1)[1]
            img_bytes = base64.b64decode(image_data)
        else:
            raise ValueError("Base64 image data must be a string")
    elif source_type == 'bytes':
        img_bytes = image_data
    else:
        raise ValueError("Unknown source_type. Must be 'base64' or 'bytes'")
    
    # Преобразуем изображение и гарантируем формат JPEG
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert("RGB")  # Убедимся, что изображение в RGB
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        img_bytes = buffered.getvalue()
    except Exception as e:
        raise ValueError(f"Failed to process image data: {e}")
    
    # Создаем уникальное имя файла
    timestamp = int(time.time())
    unique_id = uuid.uuid4().hex
    filename = f"{timestamp}_{unique_id}.jpg"
    
    # Формируем путь в S3
    key_path = f"photo_out/generated/{unique_id}/{filename}"
    
    # Используем boto3 с настройками для S3
    session = boto3.session.Session()
    s3_client = session.client(
        's3',
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        endpoint_url=s3_endpoint_url,
        region_name=s3_region_name,
        config=boto3.session.Config(signature_version='s3')
    )
    
    # Загружаем файл в S3
    try:
        s3_client.put_object(
            Body=img_bytes,
            Bucket=s3_bucket_name,
            Key=key_path,
            ACL='public-read',
            ContentType='image/jpeg'
        )
    except Exception as e:
        raise ValueError(f"Failed to upload to S3: {e}")
    
    # Формируем и возвращаем публичный URL
    result_url = f"{s3_endpoint_url.rstrip('/')}/{s3_bucket_name}/{key_path}"
    return result_url

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
                            response_data['result_url'] = uploaded_masks[0] if uploaded_masks else None
        
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
                        response_data['result_url'] = uploaded_masks[0] if uploaded_masks else None
        
        return response_data
    except Exception as e:
        return {"error": f"Failed to process SAM results: {e}", "original_response": response_data}

def process_request(job):
    """
    Proxy any incoming job JSON to the local SD/SAM server and return its JSON response.
    Expects job['input'] to contain:
      - 'path': e.g. 'sdapi/v1/txt2img' or 'sam/sam-predict'
      - 'params': dict payload for that path
    """
    input_data = job.get('input', {})
    path = input_data.get('path')
    params = input_data.get('params', {})
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
        if lk.endswith('image') and isinstance(val, str):
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

if __name__ == '__main__':
    # Если установлена переменная LOCAL и доступны fastapi/uvicorn, запускаем локальный HTTP-сервер для теста
    if os.getenv('LOCAL') and FastAPI and uvicorn:
        app = FastAPI()
        @app.post('/run')
        def run_job(job: dict = Body(...)):
            # Если передали напрямую path/params, оборачиваем в ключ 'input'
            if 'path' in job and 'params' in job:
                job_to_process = {'input': job}
            else:
                job_to_process = job
            return process_request(job_to_process)
        uvicorn.run(app, host='0.0.0.0', port=8080)
    else:
        runpod.serverless.start({"handler": process_request}) 