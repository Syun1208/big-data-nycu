#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WORKERS=8
CONFIG=configs/xgboost/exp1.yaml

python main.py download-data \
    --config "${CONFIG}" \
    --keep-snapshot \
    "$@"

python main.py download-history \
    --config "${CONFIG}" \
    --workers "${WORKERS}" \
    "$@"
