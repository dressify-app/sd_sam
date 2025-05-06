# ======================================================================
#  BASE IMAGE  –  PyTorch 2.0.1 + CUDA 11.7 + cuDNN 8
# ======================================================================
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ----------------------------------------------------------------------
#  SYSTEM DEPS
# ----------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget curl libgl1-mesa-glx libglib2.0-0 \
        build-essential cmake ninja-build g++ python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------
#  CLONE AUTOMATIC1111 WEBUI  (нужен /sdapi)
# ----------------------------------------------------------------------
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git .

# ----------------------------------------------------------------------
#  DOWNLOAD WEIGHTS
# ----------------------------------------------------------------------
# YOLOv8‑pose
RUN curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt \
        -o /app/yolov8x-pose.pt
# Segment‑Anything (ViT‑B)
RUN curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
        -o /app/sam_vit_b_01ec64.pth
# Stable‑Diffusion v1.5
RUN mkdir -p models/Stable-diffusion && \
    curl -L https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors \
        -o models/Stable-diffusion/v1-5-pruned-emaonly.safetensors

# ----------------------------------------------------------------------
#  PYTHON DEPS
# ----------------------------------------------------------------------
RUN pip install --no-cache-dir \
        ultralytics==8.2.0 \
        git+https://github.com/facebookresearch/segment-anything.git \
        opencv-python-headless \
        pillow torchvision \
        runpod boto3 requests \
        xformers==0.0.22 triton

# ----------------------------------------------------------------------
#  APP FILES
# ----------------------------------------------------------------------
COPY function_handler.py /app/function_handler.py
COPY start.sh           /app/start.sh
RUN chmod +x /app/start.sh

# ----------------------------------------------------------------------
#  CUDA & PYTORCH ENV
# ----------------------------------------------------------------------
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ENV CUDA_LAUNCH_BLOCKING=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONUNBUFFERED=1

EXPOSE 7860
ENTRYPOINT ["/bin/bash", "-c", "/app/start.sh"]
