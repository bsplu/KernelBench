#!/usr/bin/env bash
set -euo pipefail

# Analyze eval results and compare against baseline.
#
# Usage:
#   RUN_NAME=dsv32_L1_S1_T0_api LEVEL=1 HARDWARE=A5000 BASELINE=baseline_time_torch \
#   bash scripts/run_analysis.sh
#
# Optional env overrides:
#   PYTHON_BIN=/path/to/python
#   EVAL_RESULTS_DIR=runs
#   BASELINE_FILE=results/timing/A5000/baseline_time_torch.json
#   OUTPUT_FILE=runs/dsv32_L1_S1_T0_api/analysis.json

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_NAME="${RUN_NAME:-}"
LEVEL="${LEVEL:-}"
HARDWARE="${HARDWARE:-A5000}"
BASELINE="${BASELINE:-baseline_time_torch}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-}"
BASELINE_FILE="${BASELINE_FILE:-}"
OUTPUT_FILE="${OUTPUT_FILE:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -z "${RUN_NAME}" || -z "${LEVEL}" ]]; then
  echo "ERROR: RUN_NAME and LEVEL are required." >&2
  echo "Example: RUN_NAME=dsv32_L1_S1_T0_api LEVEL=1 HARDWARE=A5000 BASELINE=baseline_time_torch bash scripts/run_analysis.sh" >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" scripts/benchmark_eval_analysis.py
  "run_name=${RUN_NAME}"
  "level=${LEVEL}"
  "hardware=${HARDWARE}"
  "baseline=${BASELINE}"
)

if [[ -n "${EVAL_RESULTS_DIR}" ]]; then
  cmd+=("eval_results_dir=${EVAL_RESULTS_DIR}")
fi
if [[ -n "${BASELINE_FILE}" ]]; then
  cmd+=("baseline_file=${BASELINE_FILE}")
fi
if [[ -n "${OUTPUT_FILE}" ]]; then
  cmd+=("output_file=${OUTPUT_FILE}")
fi

echo "=== Analysis Config ==="
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "RUN_NAME=${RUN_NAME}"
echo "LEVEL=${LEVEL}"
echo "HARDWARE=${HARDWARE}"
echo "BASELINE=${BASELINE}"
[[ -n "${EVAL_RESULTS_DIR}" ]] && echo "EVAL_RESULTS_DIR=${EVAL_RESULTS_DIR}"
[[ -n "${BASELINE_FILE}" ]] && echo "BASELINE_FILE=${BASELINE_FILE}"
[[ -n "${OUTPUT_FILE}" ]] && echo "OUTPUT_FILE=${OUTPUT_FILE}"
echo "======================="

"${cmd[@]}"
