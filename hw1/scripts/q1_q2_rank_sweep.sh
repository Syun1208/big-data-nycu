#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=python
RUN_DATE=$(date +%F)
OUTPUT_DIR=outputs/${RUN_DATE}_q1_q2_rank_sweep
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > best {best = $2; selected = $1} END {print selected}')

cd "${PROJECT_ROOT}"
"${PYTHON}" main.py finetune \
    --run 16 \
    --run 64 \
    --run 256 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"

"${PYTHON}" main.py report \
    --results "${OUTPUT_DIR}/benchmark_results.csv" \
    --output-dir "${OUTPUT_DIR}/report"
