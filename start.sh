#!/bin/bash

# Вывод диагностической информации
echo "===== Diagnostics ====="
echo "PATH=$PATH"
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Проверка наличия необходимых модулей Python (оптимизировано)
echo "===== Checking Python modules ====="
python -c "import runpod; print(f'runpod version: {runpod.__version__}')" 2>/dev/null || pip install runpod
python -c "import boto3; print(f'boto3 version: {boto3.__version__}')" 2>/dev/null || pip install boto3
python -c "import requests; print(f'requests version: {requests.__version__}')" 2>/dev/null || pip install requests
python -c "import PIL; print(f'PIL version: {PIL.__version__}')" 2>/dev/null || pip install Pillow

# Проверка наличия моделей SAM
echo "===== Checking SAM models ====="
SAM_DIR="models/sam"
mkdir -p "$SAM_DIR"

# Загрузка модели SAM, если она отсутствует
SAM_MODEL="$SAM_DIR/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_MODEL" ]; then
    echo "Downloading SAM model to $SAM_MODEL..."
    curl -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -o "$SAM_MODEL"
    if [ $? -eq 0 ]; then
        echo "SAM model downloaded successfully!"
    else
        echo "ERROR: Failed to download SAM model!"
        exit 1
    fi
else
    echo "SAM model already exists at $SAM_MODEL"
fi

# Проверка переменных окружения для S3
echo "===== Checking S3 environment variables ====="
for var in S3_ACCESS_KEY S3_SECRET_KEY S3_ENDPOINT_URL S3_BUCKET_NAME S3_REGION_NAME; do
    if [ -z "${!var}" ]; then
        echo "WARNING: $var is not set or empty"
    else
        echo "$var is set"
    fi
done

# Создание функции проверки статуса процессов
check_process_status() {
    local process_name=$1
    local pid=$2
    if kill -0 $pid 2>/dev/null; then
        echo "$process_name (PID $pid) is running"
    else
        echo "WARNING: $process_name (PID $pid) has stopped"
    fi
}

# Создание файла с функцией trap для очистки
cleanup() {
    echo "Stopping background processes..."
    jobs -p | xargs -r kill
    exit 0
}

# Регистрация функции cleanup для обработки сигналов завершения
trap cleanup SIGINT SIGTERM EXIT

# Запуск приложений
echo "===== Starting applications ====="

# Запуск RunPod handler в фоне сразу (параллельно с WebUI)
echo "1. Launching RunPod handler in background..."
python function_handler.py &
HANDLER_PID=$!

echo "2. Launching WebUI with arguments: --api --listen --xformers --port 7860 --skip-torch-cuda-test --no-half-vae --no-hashing --skip-version-check --no-download-sd-model"
python launch.py --api --listen --xformers --port 7860 --skip-torch-cuda-test --no-half-vae --no-hashing --skip-version-check --no-download-sd-model &
WEBUI_PID=$!

# Функция проверки готовности API
check_api() {
    curl -s --head --fail http://127.0.0.1:7860/ >/dev/null
    return $?
}

# Wait until WebUI is available
echo "Waiting for WebUI to start..."
MAX_ATTEMPTS=300  # 5 минут (300 * 1 секунда)
ATTEMPT=0

until check_api; do
    ATTEMPT=$((ATTEMPT+1))
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo "ERROR: Timed out waiting for WebUI to start after 5 minutes!"
        echo "WebUI log (last 50 lines):"
        tail -n 50 /tmp/webui.log 2>/dev/null || echo "No log file found."
        exit 1
    fi
    echo "WebUI not ready yet (attempt $ATTEMPT of $MAX_ATTEMPTS). Sleeping for 1 second..."
    sleep 1
done

echo "WebUI is up and running!"

# Проверка запуска процессов
echo "Checking process status..."
check_process_status "WebUI" $WEBUI_PID
check_process_status "RunPod handler" $HANDLER_PID

echo "All processes started. Waiting for them to complete..."
wait 