#!/usr/bin/env bash
# 示例：在填写下方 USER 变量后执行；路径均相对于本仓库 safety-editing 根目录。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# USER: SafeEdit 安全分类器、数据集目录、指标输出目录
SAFETY_CLASSIFIER_DIR=""
DATA_DIR=""
METRICS_DIR=""

python "${ROOT}/scripts/run_safety_editing.py" \
  --editing_method FINE \
  --edited_model qwen3-14b-instruct \
  --hparams_dir "${ROOT}/hparams/fine_qwen3-14b-instruct.yaml" \
  --safety_classifier_dir "${SAFETY_CLASSIFIER_DIR}" \
  --data_dir "${DATA_DIR}" \
  --metrics_save_dir "${METRICS_DIR}" \
