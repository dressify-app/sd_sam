#!/usr/bin/env bash
set -e
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

echo "===== Diagnostics ====="
python -V
pip -V
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ------------------------------------------------------------
#  быстрая проверка весов
# ------------------------------------------------------------
MOBILE_SHA=3d3a5bb4424eb0f76fbd6d4259f1e13d6e7b826dc6e324d4b7f6bb51f3f9f08
if ! echo "${MOBILE_SHA}  /app/mobile_sam.pt" | sha256sum -c -; then
  echo "Re‑downloading mobile_sam.pt…"
  curl -L --retry 5 --retry-max-time 120 \
       https://huggingface.co/ChaoningZhang/MobileSAM/resolve/main/mobile_sam.pt \
       -o /app/mobile_sam.pt
fi

for f in /app/yolov8x-pose.pt /app/mobile_sam.pt; do
  [[ -f "$f" ]] || { echo "ERROR: weight $f not found!"; exit 1; }
done
echo "Model weights are present and verified."

# ------------------------------------------------------------
#  запускаем WebUI (только API)
# ------------------------------------------------------------
echo "Starting Stable‑Diffusion WebUI (API only)…"
export COMMANDLINE_ARGS="\
  --api --listen --port 7860 \
  --xformers \
  --skip-version-check --skip-torch-cuda-test \
  --no-hashing --disable-safe-unpickle \
  --disable-console-progressbars \
  --ckpt /app/models/Stable-diffusion/v1-5-pruned-emaonly-fp16.safetensors \
  --no-download-sd-model"
python launch.py $COMMANDLINE_ARGS > /tmp/webui.log 2>&1 &

WEBUI_PID=$!

# ------------------------------------------------------------
#  ждём, пока /sdapi/v1/txt2img станет доступен
# ------------------------------------------------------------
for i in {1..60}; do
  if curl -s -o /dev/null http://127.0.0.1:7860/sdapi/v1/txt2img; then
    echo "WebUI is ready (after $i checks)."
    break
  fi
  echo "Waiting WebUI… ($i/60)"; sleep 2
done

if ! kill -0 $WEBUI_PID 2>/dev/null; then
  echo "ERROR: WebUI process crashed!"
  tail -n 50 /tmp/webui.log || true
  exit 1
fi

# ------------------------------------------------------------
#  запускаем наш RunPod handler
# ------------------------------------------------------------
echo "Starting RunPod handler…"
python function_handler.py &
HANDLER_PID=$!

# ------------------------------------------------------------
#  trap для корректного завершения
# ------------------------------------------------------------
cleanup() {
  echo "Stopping processes…"
  kill $HANDLER_PID $WEBUI_PID 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM EXIT

wait $HANDLER_PID