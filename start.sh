#!/bin/bash

# Установка переменных для пропуска проверок репозиториев
export SKIP_PREPARE_ENVIRONMENT=1
export TORCH_COMMAND="pip install torch torchvision"
export COMMANDLINE_ARGS="--api --listen --xformers --port 7860 --skip-torch-cuda-test --no-half-vae --no-hashing --skip-version-check --no-download-sd-model --disable-git-updates --skip-python-version-check"

# Разогрев CUDA перед запуском
echo "===== Warming up CUDA ====="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); torch.ones(1).cuda()" &> /dev/null

echo "===== Diagnostics ====="
echo "PATH=$PATH"
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Проверка наличия необходимых модулей Python (только информационно)
echo "===== Checking Python modules ====="
for module in runpod boto3 requests PIL; do
    if python -c "import $module" >/dev/null 2>&1; then
        python -c "import $module; print(f'$module version: {${module}.__version__}')" 2>/dev/null || echo "$module is installed (version unknown)"
    else
        echo "$module not found, installing..."
        if [ "$module" = "PIL" ]; then
            pip install --no-cache-dir Pillow
        else
            pip install --no-cache-dir $module
        fi
    fi
done

# Проверка наличия моделей SAM (только информационно)
echo "===== Checking SAM models ====="
if [ -f "models/sam/sam_vit_h_4b8939.pth" ]; then
    echo "SAM model already exists at models/sam/sam_vit_h_4b8939.pth"
else
    echo "WARNING: SAM model missing - this shouldn't happen in the container"
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

# Создание функции trap для очистки
cleanup() {
    echo "Stopping background processes..."
    jobs -p | xargs -r kill
    exit 0
}

# Регистрация функции cleanup для обработки сигналов завершения
trap cleanup SIGINT SIGTERM EXIT

# Запуск приложений
echo "===== Starting applications ====="
echo "1. Launching WebUI in background with arguments: $COMMANDLINE_ARGS"
python launch.py $COMMANDLINE_ARGS &
WEBUI_PID=$!

# Функция проверки готовности API с более быстрым таймаутом
check_api() {
    curl -s --connect-timeout 0.5 --head --fail http://127.0.0.1:7860/ >/dev/null
    return $?
}

# Wait until WebUI is available
echo "Waiting for WebUI to start..."
MAX_ATTEMPTS=120
ATTEMPT=0

until check_api; do
    ATTEMPT=$((ATTEMPT+1))
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo "ERROR: Timed out waiting for WebUI to start!"
        exit 1
    fi
    echo "WebUI not ready yet (attempt $ATTEMPT of $MAX_ATTEMPTS). Sleeping for 2 seconds..."
    sleep 2
done

echo "WebUI is up and running!"

echo "2. Launching RunPod handler..."
python function_handler.py &
HANDLER_PID=$!

# Проверка запуска процессов
echo "Checking process status after 5 seconds..."
sleep 5
if kill -0 $WEBUI_PID 2>/dev/null; then
    echo "WebUI (PID $WEBUI_PID) is running"
else 
    echo "WARNING: WebUI (PID $WEBUI_PID) has stopped"
fi

if kill -0 $HANDLER_PID 2>/dev/null; then
    echo "RunPod handler (PID $HANDLER_PID) is running"
else
    echo "WARNING: RunPod handler (PID $HANDLER_PID) has stopped"
fi

echo "All processes started. Waiting for them to complete..."
wait 