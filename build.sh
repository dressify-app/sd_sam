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

# Сборка образа с использованием NVIDIA runtime
echo "Building image: $IMAGE_NAME:$TAG"
docker build --runtime=nvidia -t $IMAGE_NAME:$TAG .

# Проверка результата сборки
if [ $? -eq 0 ]; then
    echo "===== Build successful! ====="
    echo "Image $IMAGE_NAME:$TAG is ready"
    echo ""
    echo "You can run the container with:"
    echo "docker run --gpus all -p 7860:7860 $IMAGE_NAME:$TAG"
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