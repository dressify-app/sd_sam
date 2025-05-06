#!/usr/bin/env bash
set -e

echo "===== Diagnostics ====="
python -V
pip -V
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ------------------------------------------------------------
#  быстрая проверка весов
# ------------------------------------------------------------
for f in /app/yolov8x-pose.pt /app/mobile_sam.pt; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: weight $f not found in container!"
    exit 1
  fi
done
echo "Model weights are present."

# ------------------------------------------------------------
#  запускаем WebUI (только API)
# ------------------------------------------------------------
echo "Starting Stable‑Diffusion WebUI (API only)…"
python launch.py \
  --api --listen --port 7860 \
  --xformers \
  --skip-version-check \
  --no-download-sd-model \
  > /tmp/webui.log 2>&1 &

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