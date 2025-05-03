#!/bin/bash

# Вывод информации о команде сборки
echo "===== Building Docker image with NVIDIA runtime ====="
echo "This script builds the Docker image using NVIDIA runtime to properly compile CUDA extensions"

# Проверка наличия Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH!"
    exit 1
fi

# Переменные
IMAGE_NAME="sd-sam-segmentation"
TAG="latest"

# Проверка аргументов
USE_NVIDIA_RUNTIME=true
CPU_ONLY=false

for arg in "$@"; do
    case $arg in
        --cpu-only)
            CPU_ONLY=true
            USE_NVIDIA_RUNTIME=false
            shift
            ;;
        --no-nvidia-runtime)
            USE_NVIDIA_RUNTIME=false
            shift
            ;;
    esac
done

# Сборка образа с соответствующими параметрами
echo "Building image: $IMAGE_NAME:$TAG"

BUILD_ARGS=""
if [ "$CPU_ONLY" = true ]; then
    BUILD_ARGS="--build-arg DISABLE_CUDA=1"
    echo "Building for CPU-only mode"
fi

if [ "$USE_NVIDIA_RUNTIME" = true ]; then
    echo "Using NVIDIA runtime for build"
    docker build --runtime=nvidia $BUILD_ARGS -t $IMAGE_NAME:$TAG .
else
    echo "Using standard runtime for build"
    docker build $BUILD_ARGS -t $IMAGE_NAME:$TAG .
fi

# Проверка результата сборки
if [ $? -eq 0 ]; then
    echo "===== Build successful! ====="
    echo "Image $IMAGE_NAME:$TAG is ready"
    echo ""
    echo "You can run the container with:"
    if [ "$CPU_ONLY" = true ]; then
        echo "docker run -p 7860:7860 -e DISABLE_CUDA=1 $IMAGE_NAME:$TAG"
    else
        echo "docker run --gpus all -p 7860:7860 $IMAGE_NAME:$TAG"
    fi
else
    echo "===== Build failed! ====="
    echo "If you encounter CUDA compilation issues, make sure:"
    echo "1. NVIDIA Container Toolkit is installed (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)"
    echo "2. Your Docker configuration has NVIDIA runtime enabled"
    echo ""
    echo "Alternative build command if --runtime=nvidia is not recognized:"
    echo "DOCKER_BUILDKIT=1 docker build --build-arg BUILDKIT_INLINE_CACHE=1 -t $IMAGE_NAME:$TAG ."
    exit 1
fi 