#!/usr/bin/env bash
# End-to-end check on macOS (Apple Silicon): venv, lint, tests, CPU+CoreML benchmark, report.
# Usage: scripts/mac_check.sh [--full]   (--full = 200 iters + 3 s throughput, default is quick)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in python3.13 python3.12 python3.11; do command -v "$c" >/dev/null && PY="$c" && break; done
fi
[[ -n "$PY" ]] || { echo "Python 3.11+ not found (brew install python@3.12)"; exit 1; }
echo "==> using $($PY --version) ($PY)"

[[ -d .venv ]] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip --version >/dev/null 2>&1 || python -m ensurepip >/dev/null
python -m pip install -q --upgrade pip
python -m pip install -q -e ".[dev]"

echo "==> lint + tests"
ruff check .
ruff format --check .
pytest -q

echo "==> execution providers"
edge-ai-bench providers

ITERS=50; SECS=1
if [[ "${1:-}" == "--full" ]]; then ITERS=200; SECS=3; fi
OUT="results/mac-$(sysctl -n machdep.cpu.brand_string | tr ' ' '-' | tr '[:upper:]' '[:lower:]').json"
echo "==> benchmark (cpu, coreml, qnn/cuda are skipped if absent) -> $OUT"
edge-ai-bench run --providers cpu,coreml,qnn,cuda --iters "$ITERS" --seconds "$SECS" --out "$OUT"

echo "==> report"
edge-ai-bench report results/*.json -o report.html
edge-ai-bench aihub setup | head -3
echo "OK - open report.html"
