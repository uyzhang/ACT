#!/usr/bin/env bash
# ACT inference on Qwen3-VL-8B-Instruct, 32 frames, retain_ratio=0.25.
set -euo pipefail

# shellcheck source=/dev/null
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/_port_utils.sh"

pretrained_model="qwen3_vl"
pretrained="${pretrained:-Qwen/Qwen3-VL-8B-Instruct}"
method="act"

TASK="${TASK:-videomme}"

model_args="pretrained=${pretrained},max_num_frames=32,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

BASE_LOG_DIR="logs/${pretrained_model}/32/${method}"
mkdir -p "${BASE_LOG_DIR}"

ts=$(date +"%m-%d-%H-%M-%S")
TAG="rr${ACT_RETAIN_RATIO:-0.25}_pa${ACT_PROJECT_ALPHA:-0.3}_g${ACT_GREEDY_COVERAGE:-1}"
JOB_NAME="${pretrained_model}_${method}_${TASK}_${TAG}"
output_path="${BASE_LOG_DIR}/${ts}_${JOB_NAME}"
log_file="${BASE_LOG_DIR}/${ts}_${JOB_NAME}.log"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
method="${method}" \
ACT_RETAIN_RATIO="${ACT_RETAIN_RATIO:-0.25}" \
ACT_MIN_K="${ACT_MIN_K:-1}" \
ACT_BUDGET_TEMP="${ACT_BUDGET_TEMP:-0.7}" \
ACT_NUM_REGIONS_H="${ACT_NUM_REGIONS_H:-4}" \
ACT_NUM_REGIONS_W="${ACT_NUM_REGIONS_W:-4}" \
ACT_SINKHORN_EPSILON="${ACT_SINKHORN_EPSILON:-0.05}" \
ACT_SINKHORN_ITERS="${ACT_SINKHORN_ITERS:-50}" \
ACT_CURV_WEIGHT_BETA="${ACT_CURV_WEIGHT_BETA:-1.0}" \
ACT_PROJECT_ALPHA="${ACT_PROJECT_ALPHA:-0.3}" \
ACT_SAME_REGION="${ACT_SAME_REGION:-1}" \
ACT_GREEDY_COVERAGE="${ACT_GREEDY_COVERAGE:-1}" \
ACT_GREEDY_PRE_RATIO="${ACT_GREEDY_PRE_RATIO:-2.0}" \
run_with_port_retry accelerate launch \
  --num_processes="${NUM_PROCESSES}" \
  -m lmms_eval \
  --model "${pretrained_model}" \
  --model_args "${model_args}" \
  --tasks "${TASK}" \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix "${TAG}" \
  --output_path "${output_path}" 2>&1 | tee "${log_file}"
