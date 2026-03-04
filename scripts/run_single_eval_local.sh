#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_single_eval_local.sh
#   bash scripts/run_single_eval_local.sh level=2 problem_id=40 model_name=deepseek-ai/DeepSeek-V3.2
#
# Optional env:
#   PYTHON_BIN=/path/to/python
#   CLEAR_TORCH_EXTENSIONS=1

PYTHON_BIN="${PYTHON_BIN:-python3}"
CLEAR_TORCH_EXTENSIONS="${CLEAR_TORCH_EXTENSIONS:-0}"

# module purge
module load nvhpc/23.5
module load gcc/11

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

# Force torch extension builds to use GNU toolchain (avoid nvcc -> nvc)
GPP_BIN="$(command -v g++)"
GCC_BIN="$(command -v gcc)"
if [[ -z "${GPP_BIN}" || -z "${GCC_BIN}" ]]; then
  echo "ERROR: gcc/g++ not found after module load gcc/11" >&2
  exit 1
fi
# Torch's CUDA extension path often feeds CC into nvcc -ccbin; use g++ for C++ host compilation.
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

# For NVHPC, nvcc is under .../compilers/bin, but CUDA toolkit libs live under .../cuda/<ver>.
if [[ "${NVCC_ROOT}" == */compilers ]] && [[ -n "${NVHPC_ROOT:-}" ]]; then
  CUDA_VER="$(nvcc --version | awk -F'release ' '/release/{print $2}' | awk -F',' '{print $1}' | tr -d '[:space:]')"
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
# Add both CUDA runtime libs and math libs (cublas) for NVHPC layout.
if [[ -n "${NVHPC_ROOT:-}" && -d "${NVHPC_ROOT}/math_libs/${CUDA_VER:-}/lib64" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${NVHPC_ROOT}/math_libs/${CUDA_VER}/lib64:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64:${NVHPC_ROOT}/math_libs/${CUDA_VER}/lib64:${LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
fi

if [[ "${CLEAR_TORCH_EXTENSIONS}" == "1" ]]; then
  rm -rf "${HOME}/.cache/torch_extensions"
fi

echo "=== Build Environment ==="
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "nvcc=$(command -v nvcc)"
echo "gcc=$(command -v gcc)"
echo "g++=$(command -v g++)"
echo "CUDA_HOME=${CUDA_HOME}"
echo "NVHPC_ROOT=${NVHPC_ROOT:-}"
echo "CC=${CC}"
echo "CXX=${CXX}"
echo "CUDAHOSTCXX=${CUDAHOSTCXX}"
echo "LIBRARY_PATH=${LIBRARY_PATH}"
echo "========================="

if [[ "$#" -eq 0 ]]; then
  set -- \
    dataset_src=local \
    level=2 \
    problem_id=40 \
    server_type=local \
    model_name=deepseek-ai/DeepSeek-V3.2 \
    backend=cuda \
    gpu_arch=Ampere \
    log=true \
    log_generated_kernel=true \
    log_eval_result=true
fi

"${PYTHON_BIN}" scripts/generate_and_eval_single_sample.py "$@"
