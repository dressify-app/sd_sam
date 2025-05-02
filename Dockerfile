# Используем официальный образ Python 3.10
FROM python:3.10-slim

# Установка необходимых системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Клонирование репозитория WebUI
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git .

# Клонируем плагин Segment-Anything
RUN git clone https://github.com/continue-revolution/sd-webui-segment-anything.git extensions/segment-anything

# Устанавливаем зависимости для SAM и WebUI
RUN pip install --no-cache-dir segment-anything pillow torchvision
RUN pip install --no-cache-dir runpod boto3 requests

# Копируем наши файлы
COPY function_handler.py /app/function_handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Экспортируем порт для API
EXPOSE 7860

# Запускаем скрипт запуска
ENTRYPOINT ["/bin/bash", "-c", "/app/start.sh"] 