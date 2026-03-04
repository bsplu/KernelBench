#!/usr/bin/env bash
set -euo pipefail

max_loop=5

export KB_DEBUG_COT=0

# Retries per (level, sample) group until all kernels exist.
# for sample in 1 10; do
for sample in 1; do
  if [[ "${sample}" -eq 1 ]]; then
    temperature=0.0
  else
    temperature=1.6
  fi
  # temperature=0.0
  for level in 1 2 3; do
  # for level in 3; do
    run_name="dsv32_L${level}_S${sample}"
    echo "=== Start run_name=${run_name} (level=${level}, num_samples=${sample}) ==="

    completed=false
    for i in $(seq 1 "${max_loop}"); do
      echo "[${run_name}] attempt ${i}/${max_loop} @ $(date '+%F %T')"
      tmp_log="$(mktemp)"

      # Keep output on screen and in temp log for completion detection.
      set +e
      python scripts/generate_samples.py \
        run_name="${run_name}" \
        dataset_src=local \
        level="${level}" \
        num_workers=20 \
        num_samples="${sample}" \
        server_type=local \
        model_name="deepseek-v3_2-chat" \
        temperature=${temperature} \
        custom_prompt_key=compile_safe \
        is_reasoning_model=true \
        enable_self_review=true \
        generation_retries=2 \
        include_hardware_info=true \
        hardware_gpu_name="A5000" \
        log_model_output=true \
        backend=cuda 2>&1 | tee "${tmp_log}"
      cmd_status=${PIPESTATUS[0]}
      set -e

      # Done condition from generate_samples.py output.
      if grep -Eq "All [0-9]+ kernels already exist" "${tmp_log}"; then
        echo "[${run_name}] complete: all kernels exist."
        completed=true
        rm -f "${tmp_log}"
        break
      fi

      rm -f "${tmp_log}"

      if [[ ${cmd_status} -ne 0 ]]; then
        echo "[${run_name}] warning: command exit code=${cmd_status}, will retry."
      else
        echo "[${run_name}] not complete yet, retrying."
      fi
    done

    if [[ "${completed}" != true ]]; then
      echo "[${run_name}] ERROR: not complete after ${max_loop} attempts."
      # exit 1
    fi

    echo "=== Finished run_name=${run_name} ==="
  done
done

# model_name="deepseek-ai/DeepSeek-V3.2"
# model_name=deepseek-v3_2-chat \
# custom_prompt_key=compile_safe \
# subset=1,10 \
# prompt_option=one_shot \
# "MiniMax-M2.5
