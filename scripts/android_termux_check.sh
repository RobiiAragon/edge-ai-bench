#!/data/data/com.termux/files/usr/bin/bash
# Run edge-ai-bench on Android via Termux (tested target: Galaxy Tab S9 Ultra, Snapdragon 8 Gen 2).
# Termux's onnxruntime has CPU, XNNPACK and NNAPI EPs (no QNN). All three are benchmarked;
# NNAPI may route parts of the graph to the Hexagon NPU/GPU through Samsung's driver.
#
#   pkg install -y git && git clone https://github.com/RobiiAragon/edge-ai-bench
#   cd edge-ai-bench && bash scripts/android_termux_check.sh
#
# Then copy results/*.json back to the Mac and run `edge-ai-bench report results/*.json`.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${PREFIX:-}" || "$PREFIX" != *com.termux* ]]; then
  echo "This script is meant for Termux on Android."; exit 1
fi

echo "==> Termux packages (prebuilt numpy/onnxruntime; pip wheels do not exist for Android)"
pkg update -y >/dev/null
pkg install -y python python-numpy python-onnxruntime git >/dev/null || {
  echo "python-onnxruntime not available from 'pkg'. Try: pkg install tur-repo && pkg install python-onnxruntime"
  exit 1
}

python - <<'PY'
import sys, numpy, onnxruntime as ort
assert sys.version_info >= (3, 11), sys.version
print("python", sys.version.split()[0], "| numpy", numpy.__version__,
      "| onnxruntime", ort.__version__, ort.get_available_providers())
PY

echo "==> install edge-ai-bench (no deps: uses the Termux numpy/onnxruntime)"
python -m pip install -q --no-deps --no-build-isolation -e . 2>/dev/null \
  || python -m pip install -q --no-deps -e .

echo "==> device"
edge-ai-bench info
edge-ai-bench providers

ITERS="${ITERS:-200}"
SECS="${SECS:-3}"
OUT="results/android-$(getprop ro.product.model | tr ' ' '-' | tr '[:upper:]' '[:lower:]').json"
echo "==> benchmark (cpu, xnnpack, nnapi; $ITERS iters) -> $OUT"
# Keep the screen on and the tablet plugged in; thermal throttling skews long runs.
edge-ai-bench run --providers "${PROVIDERS:-cpu,xnnpack,nnapi}" --iters "$ITERS" --seconds "$SECS" --out "$OUT"

echo "==> done. Share $OUT (e.g. termux-setup-storage && cp $OUT ~/storage/downloads/)"
