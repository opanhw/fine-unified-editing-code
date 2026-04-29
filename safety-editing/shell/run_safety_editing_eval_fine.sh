#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# USER: 同上，填入本机路径
SAFETY_CLASSIFIER_DIR=""
DATA_DIR=""
METRICS_DIR=""

python "${ROOT}/scripts/run_safety_editing_eval.py" \
  --editing_method FINE \
  --edited_model qwen3-14b-instruct \
  --hparams_dir "${ROOT}/hparams/fine_qwen3-14b-instruct.yaml" \
  --safety_classifier_dir "${SAFETY_CLASSIFIER_DIR}" \
  --data_dir "${DATA_DIR}" \
  --metrics_save_dir "${METRICS_DIR}" \
  --eval_index 10
