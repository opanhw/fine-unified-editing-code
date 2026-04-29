#!/usr/bin/env bash
# USER: 安装 OpenCompass 后，确保 opencompass 在 PATH 中；按需设置 GPU
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
opencompass "${ROOT}/scripts/opencompass_eval.py"
