#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# USER: 填入本机路径
SAFETY_CLASSIFIER_DIR=""
DATA_DIR=""
METRICS_DIR=""

python "${ROOT}/scripts/run_safety_editing_pre.py" \
  --editing_method DINM \
  --edited_model qwen3-14b-instruct \
  --hparams_dir "${ROOT}/hparams/dinm_qwen3-14b-instruct.yaml" \
  --safety_classifier_dir "${SAFETY_CLASSIFIER_DIR}" \
  --data_dir "${DATA_DIR}" \
  --metrics_save_dir "${METRICS_DIR}"
