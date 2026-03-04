#!/usr/bin/env bash
set -euo pipefail

# Batch evaluate generated kernels in runs/<run_name> from run_inf.sh
#
# Usage:
#   bash scripts/run_eval.sh
#
# Optional env overrides:
#   PYTHON_BIN=/path/to/python
#   MAX_LOOP=2
#   NUM_GPU_DEVICES=1
#   TIMEOUT=300
#   BACKEND=cuda
#   PRECISION=fp32
#   CLEAR_TORCH_EXTENSIONS=0

PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_LOOP="${MAX_LOOP:-1}"
NUM_GPU_DEVICES="${NUM_GPU_DEVICES:-1}"
TIMEOUT="${TIMEOUT:-300}"
BACKEND="${BACKEND:-cuda}"
PRECISION="${PRECISION:-fp32}"
CLEAR_TORCH_EXTENSIONS="${CLEAR_TORCH_EXTENSIONS:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

module load nvhpc/23.5
module load gcc/11

# Build env for torch CUDA extension compile
GPP_BIN="$(command -v g++)"
if [[ -z "${GPP_BIN}" ]]; then
  echo "ERROR: g++ not found after module load gcc/11" >&2
  exit 1
fi
export CC="${GPP_BIN}"
export CXX="${GPP_BIN}"
export CUDAHOSTCXX="${GPP_BIN}"
unset CCBIN || true

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found after module load nvhpc/23.5" >&2
  exit 1
fi

NVCC_BIN="$(command -v nvcc)"
NVCC_ROOT="$(dirname "$(dirname "${NVCC_BIN}")")"
CUDA_VER="$(nvcc --version | awk -F'release ' '/release/{print $2}' | awk -F',' '{print $1}' | tr -d '[:space:]')"

if [[ "${NVCC_ROOT}" == */compilers ]] && [[ -n "${NVHPC_ROOT:-}" ]]; then
  CUDA_HOME_CANDIDATE="${NVHPC_ROOT}/cuda/${CUDA_VER}"
  if [[ -d "${CUDA_HOME_CANDIDATE}" ]]; then
    export CUDA_HOME="${CUDA_HOME_CANDIDATE}"
  else
    export CUDA_HOME="${NVCC_ROOT}"
  fi
else
  export CUDA_HOME="${NVCC_ROOT}"
fi

export PATH="${CUDA_HOME}/bin:${PATH}"
if [[ -n "${NVHPC_ROOT:-}" && -d "${NVHPC_ROOT}/math_libs/${CUDA_VER}/lib64" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${NVHPC_ROOT}/math_libs/${CUDA_VER}/lib64:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64:${NVHPC_ROOT}/math_libs/${CUDA_VER}/lib64:${LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
fi

if [[ "${CLEAR_TORCH_EXTENSIONS}" == "1" ]]; then
  rm -rf "${HOME}/.cache/torch_extensions"
fi

echo "=== Eval Environment ==="
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "nvcc=$(command -v nvcc)"
echo "CUDA_HOME=${CUDA_HOME}"
echo "CC=${CC}"
echo "CXX=${CXX}"
echo "CUDAHOSTCXX=${CUDAHOSTCXX}"
echo "NUM_GPU_DEVICES=${NUM_GPU_DEVICES}"
echo "TIMEOUT=${TIMEOUT}"
echo "BACKEND=${BACKEND}"
echo "PRECISION=${PRECISION}"
echo "========================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# for sample in 1 10; do
for sample in 1; do
  for level in 1 2 3; do
  # for level in 1; do
    # if [[ "${level}" -ge 3 ]]; then
    #   num_correct_trials=1
    # else
      num_correct_trials=5
    # fi
    run_name="dsv32_L${level}_S${sample}"
    # run_name="codex_debug"
    run_dir="runs/${run_name}"

    if [[ ! -d "${run_dir}" ]]; then
      echo "[${run_name}] WARNING: ${run_dir} not found, skip."
      continue
    fi

    echo "=== Start eval run_name=${run_name} (level=${level}, samples=${sample}) ==="
    completed=false

    for i in $(seq 1 "${MAX_LOOP}"); do
      echo "[${run_name}] eval attempt ${i}/${MAX_LOOP} @ $(date '+%F %T')"
      tmp_log="$(mktemp)"

      set +e
      "${PYTHON_BIN}" scripts/eval_from_generations.py \
        run_name="${run_name}" \
        dataset_src=local \
        level="${level}" \
        num_samples_per_problem="${sample}" \
        num_gpu_devices="${NUM_GPU_DEVICES}" \
        timeout="${TIMEOUT}" \
        num_correct_trials="${num_correct_trials}" \
        backend="${BACKEND}" \
        precision="${PRECISION}" \
        eval_mode=local \
        gpu_arch=Ampere 2>&1 | tee "${tmp_log}"
      cmd_status=${PIPESTATUS[0]}
      set -e

      # Done condition: no unevaluated samples left.
      if grep -Eq "Start evaluation on 0 unevaluated samples" "${tmp_log}"; then
        echo "[${run_name}] complete: no unevaluated samples left."
        completed=true
        rm -f "${tmp_log}"
        break
      fi

      rm -f "${tmp_log}"

      if [[ ${cmd_status} -ne 0 ]]; then
        echo "[${run_name}] warning: eval command exit code=${cmd_status}, retrying."
      else
        echo "[${run_name}] not complete yet, retrying."
      fi
    done

    if [[ "${completed}" != true ]]; then
      echo "[${run_name}] ERROR: eval not complete after ${MAX_LOOP} attempts."
      # exit 1
    fi
    echo "=== Finished eval run_name=${run_name} ==="
  done
done
        # subset=1,10 \
