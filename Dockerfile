# Используем образ Python 3.10 с поддержкой CUDA
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# Установка необходимых системных зависимостей (оптимизировано)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    cmake \
    ninja-build \
    g++ \
    python3-dev \
    nvidia-cuda-toolkit \
    libcupti-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Улучшенные настройки GIT для ускорения клонирования
RUN git config --global core.compression 9

# Клонирование репозитория WebUI с оптимизированными параметрами
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git .

# Клонируем плагин Segment-Anything
RUN git clone --depth 1 https://github.com/continue-revolution/sd-webui-segment-anything.git extensions/segment-anything && \
    sed -i 's/def install_goundingdino():.*/def install_goundingdino():\n    return/g' extensions/segment-anything/scripts/dino.py

# Предварительно загружаем модель GroundingDINO для избежания повторных загрузок
RUN mkdir -p extensions/segment-anything/models/grounding-dino \
    && curl -L "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth" \
        -o extensions/segment-anything/models/grounding-dino/groundingdino_swint_ogc.pth

# Предварительное клонирование дополнительных репозиториев для ускорения старта
RUN mkdir -p repositories \
    && git clone --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git repositories/stable-diffusion-webui-assets \
    && git clone --depth 1 https://github.com/Stability-AI/stablediffusion.git repositories/stable-diffusion-stability-ai \
    && git clone https://github.com/Stability-AI/generative-models.git repositories/generative-models \
    && git clone https://github.com/crowsonkb/k-diffusion.git repositories/k-diffusion \
    && git clone --depth 1 https://github.com/salesforce/BLIP.git repositories/BLIP

# Предварительная загрузка модели SAM для избежания повторных загрузок
RUN mkdir -p models/sam \
    && curl -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -o models/sam/sam_vit_h_4b8939.pth

# Предварительная загрузка основной модели Stable Diffusion
RUN mkdir -p models/Stable-diffusion \
    && curl -L "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors" \
    -o models/Stable-diffusion/v1-5-pruned-emaonly.safetensors

# Предварительная загрузка конфигурационных файлов для моделей
RUN mkdir -p configs \
    && curl -L "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml" \
    -o configs/v1-inference.yaml

# Создание директории для кэша моделей и предварительная загрузка токенов
ENV XDG_CACHE_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
RUN mkdir -p $TRANSFORMERS_CACHE && \
    pip install --no-cache-dir transformers && \
    python -c "from transformers import CLIPTokenizer; CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')" && \
    python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased')"

# Устанавливаем зависимости для SAM и WebUI с оптимизацией кэша
RUN pip install --no-cache-dir segment-anything pillow torchvision
RUN pip install --no-cache-dir runpod boto3 requests
RUN pip install --no-cache-dir --upgrade "huggingface_hub[hf_xet]" && \
    pip install --no-cache-dir hf_xet

# Оптимизация CUDA и настройки памяти
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ENV CUDA_LAUNCH_BLOCKING=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV PYTHONUNBUFFERED=1
ENV CUDA_MODULE_LOADING=LAZY

# Ускорение NVIDIA CUDA
ENV CUDA_AUTO_BOOST=1
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV NCCL_P2P_DISABLE=0

# Устанавливаем xformers с поддержкой CUDA
RUN pip install --no-cache-dir xformers==0.0.22 triton

# Установка GroundingDINO последним шагом
RUN git clone --depth 1 https://github.com/IDEA-Research/GroundingDINO.git && \
    cd GroundingDINO && \
    pip install --no-cache-dir -e . && \
    cd ..

# Копируем наши файлы
COPY function_handler.py /app/function_handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Предварительная инициализация моделей для ускорения запуска
RUN mkdir -p /app/extensions/segment-anything/models/groundingdino && \
    mkdir -p /app/extensions/segment-anything/models/sam && \
    ln -sf /app/models/sam/sam_vit_h_4b8939.pth /app/extensions/segment-anything/models/sam/ && \
    ln -sf /app/extensions/segment-anything/models/grounding-dino/groundingdino_swint_ogc.pth /app/extensions/segment-anything/models/groundingdino/

# Экспортируем порт для API
EXPOSE 7860

# Запускаем скрипт запуска
ENTRYPOINT ["/bin/bash", "-c", "/app/start.sh"] 