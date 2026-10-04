#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=python
RUN_DATE=$(date +%F)
OUTPUT_DIR=outputs/${RUN_DATE}_q4_spectrum
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7

cd "${PROJECT_ROOT}"
"${PYTHON}" main.py spectrum \
    --layers 0 7 15 \
    --energy-threshold 50 \
    --plot-limit 200 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
