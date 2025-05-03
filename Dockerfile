# Используем образ Python 3.10 с поддержкой CUDA
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

# Установка необходимых системных зависимостей (оптимизировано)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    cmake \
    ninja-build \
    libjpeg-dev \
    libpng-dev \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Клонирование репозитория WebUI
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git .

# Клонируем плагин Segment-Anything
RUN git clone https://github.com/continue-revolution/sd-webui-segment-anything.git extensions/segment-anything

# Предварительно загружаем модель GroundingDINO для избежания повторных загрузок
RUN mkdir -p extensions/segment-anything/models/grounding-dino \
    && curl -L "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth" \
    -o extensions/segment-anything/models/grounding-dino/groundingdino_swint_ogc.pth

# Предварительное клонирование дополнительных репозиториев для ускорения старта
RUN mkdir -p repositories \
    && git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git repositories/stable-diffusion-webui-assets \
    && git clone https://github.com/Stability-AI/stablediffusion.git repositories/stable-diffusion-stability-ai \
    && git clone https://github.com/Stability-AI/generative-models.git repositories/generative-models \
    && git clone https://github.com/crowsonkb/k-diffusion.git repositories/k-diffusion \
    && git clone https://github.com/salesforce/BLIP.git repositories/BLIP

# Предварительная загрузка модели SAM для избежания повторных загрузок
RUN mkdir -p models/sam \
    && curl -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -o models/sam/sam_vit_h_4b8939.pth

# Устанавливаем зависимости для SAM и WebUI
RUN pip install --no-cache-dir segment-anything pillow torchvision
RUN pip install --no-cache-dir runpod boto3 requests
# Устанавливаем GroundingDINO с принудительной компиляцией CUDA расширений
RUN pip uninstall -y groundingdino \
    && git clone https://github.com/IDEA-Research/GroundingDINO.git /tmp/GroundingDINO \
    && cd /tmp/GroundingDINO \
    && pip install torch==2.0.1 torchvision==0.15.2 \
    && apt-get update && apt-get install -y ninja-build \
    && FORCE_CUDA=1 pip install -v -e . \
    && python -c "from groundingdino.util.slconfig import SLConfig; from groundingdino.models import build_model; from groundingdino.util.utils import clean_state_dict; print('GroundingDINO import test successful')" \
    && cd /app \
    && rm -rf /tmp/GroundingDINO

# Устанавливаем xformers с поддержкой CUDA
RUN pip install --no-cache-dir xformers==0.0.22 triton

# Копируем наши файлы
COPY function_handler.py /app/function_handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Экспортируем порт для API
EXPOSE 7860

# Запускаем скрипт запуска
ENTRYPOINT ["/bin/bash", "-c", "/app/start.sh"] 