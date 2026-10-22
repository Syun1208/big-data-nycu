#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=python
STORAGE=/mnt/HDD6/longpm/big-data-nycu
PRED=${STORAGE}/outputs/hw2/submission/output_314540080.csv
TRUTH=

export CUDA_VISIBLE_DEVICES=

"${PYTHON}" -m pytest \
    "$@"

if [[ ! -f "${PRED}" ]]; then
    echo "No submission at ${PRED} yet. Run bash scripts/train.sh first."
elif [[ -n "${TRUTH}" ]]; then
    "${PYTHON}" main.py evaluate \
        --pred "${PRED}" \
        --truth "${TRUTH}"
else
    "${PYTHON}" main.py evaluate \
        --pred "${PRED}" \
        --format-only
fi

"${PYTHON}" main.py benchmark
