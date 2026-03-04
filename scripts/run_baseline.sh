#!/usr/bin/env bash
set -euo pipefail

# Generate baseline timing JSON for current hardware.
#
# Usage:
#   bash scripts/run_baseline.sh
#
# Optional env overrides:
#   PYTHON_BIN=/path/to/python
#   HARDWARE=A5000
#   PRECISION=fp32
#   BASELINE_NAME=baseline_time_torch
#   USE_TORCH_COMPILE=0
#   TORCH_COMPILE_BACKEND=inductor
#   TORCH_COMPILE_OPTIONS=default
#
# Output:
#   results/timing/${HARDWARE}/${BASELINE_NAME}.json

PYTHON_BIN="${PYTHON_BIN:-python3}"
HARDWARE="${HARDWARE:-A5000}"
PRECISION="${PRECISION:-fp32}"
BASELINE_NAME="${BASELINE_NAME:-baseline_time_torch}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-0}"
TORCH_COMPILE_BACKEND="${TORCH_COMPILE_BACKEND:-inductor}"
TORCH_COMPILE_OPTIONS="${TORCH_COMPILE_OPTIONS:-default}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "${USE_TORCH_COMPILE}" != "0" && "${USE_TORCH_COMPILE}" != "1" ]]; then
  echo "ERROR: USE_TORCH_COMPILE must be 0 or 1, got ${USE_TORCH_COMPILE}" >&2
  exit 1
fi

echo "=== Baseline Config ==="
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "HARDWARE=${HARDWARE}"
echo "PRECISION=${PRECISION}"
echo "BASELINE_NAME=${BASELINE_NAME}"
echo "USE_TORCH_COMPILE=${USE_TORCH_COMPILE}"
echo "TORCH_COMPILE_BACKEND=${TORCH_COMPILE_BACKEND}"
echo "TORCH_COMPILE_OPTIONS=${TORCH_COMPILE_OPTIONS}"
echo "======================="

"${PYTHON_BIN}" - <<PY
from scripts.generate_baseline_time import record_baseline_times

use_torch_compile = bool(int("${USE_TORCH_COMPILE}"))
torch_compile_backend = None if not use_torch_compile else "${TORCH_COMPILE_BACKEND}"
torch_compile_options = None if not use_torch_compile else "${TORCH_COMPILE_OPTIONS}"
file_name = "${HARDWARE}/${BASELINE_NAME}.json"

print(f"[Baseline] Writing to results/timing/{file_name}")
record_baseline_times(
    use_torch_compile=use_torch_compile,
    torch_compile_backend=torch_compile_backend,
    torch_compile_options=torch_compile_options,
    file_name=file_name,
    precision="${PRECISION}",
)
print("[Baseline] Done.")
PY
