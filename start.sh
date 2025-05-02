#!/bin/bash

# Вывод диагностической информации
echo "===== Diagnostics ====="
echo "PATH=$PATH"
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Проверка наличия необходимых модулей Python
echo "===== Checking Python modules ====="
for module in runpod boto3 requests PIL; do
    if python -c "import $module" >/dev/null 2>&1; then
        python -c "import $module; print(f'$module version: {${module}.__version__}')" 2>/dev/null || echo "$module is installed (version unknown)"
    else
        echo "$module not found, attempting to install..."
        if [ "$module" = "PIL" ]; then
            pip install Pillow
        else
            pip install $module
        fi
    fi
done

# Проверка наличия curl
if ! command -v curl &> /dev/null; then
    echo "curl not found, installing..."
    apt-get update && apt-get install -y curl
fi

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

# Проверка наличия SD WebUI
if [ ! -f "launch.py" ]; then
    echo "ERROR: launch.py not found in $(pwd)!"
    echo "Files in current directory:"
    ls -la
    exit 1
fi

# Проверка наличия нашего handler
if [ ! -f "function_handler.py" ]; then
    echo "ERROR: function_handler.py not found in $(pwd)!"
    echo "Files in current directory:"
    ls -la
    exit 1
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
echo "1. Launching WebUI in background with arguments: --api --listen --port 7860"
python launch.py --api --listen --port 7860 &
WEBUI_PID=$!

# Wait until WebUI is available
echo "Waiting for WebUI to start..."
MAX_ATTEMPTS=100  # ~8 минут (100 * 5 секунд)
ATTEMPT=0

until curl -s --head --fail http://127.0.0.1:7860/; do
    ATTEMPT=$((ATTEMPT+1))
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo "ERROR: Timed out waiting for WebUI to start after 5 minutes!"
        echo "WebUI log (last 50 lines):"
        tail -n 50 /tmp/webui.log 2>/dev/null || echo "No log file found."
        exit 1
    fi
    echo "WebUI not ready yet (attempt $ATTEMPT of $MAX_ATTEMPTS). Sleeping for 5 seconds..."
    sleep 5
done

echo "WebUI is up and running!"

echo "2. Launching RunPod handler..."
python function_handler.py &
HANDLER_PID=$!

# Проверка запуска процессов
echo "Checking process status after 5 seconds..."
sleep 5
check_process_status "WebUI" $WEBUI_PID
check_process_status "RunPod handler" $HANDLER_PID

echo "All processes started. Waiting for them to complete..."
wait 