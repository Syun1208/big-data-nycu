#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WORKERS=8
MIN_FREE_GB=3
CONFIG=configs/xgboost/exp1.yaml
RAW_DIR=/mnt/HDD1/longpm/big-data-nycu/gharchive
START="2025-12-01 00:00:00"
END="2026-11-21 03:00:00"

python main.py download-data \
    --config "${CONFIG}"

python main.py download-gharchive \
    --raw-dir "${RAW_DIR}" \
    --start "${START}" \
    --end "${END}" \
    --workers "${WORKERS}" \
    --min-free-gb "${MIN_FREE_GB}"

python main.py preprocess \
    --config "${CONFIG}" \
    --raw-dir "${RAW_DIR}" \
    --start "${START}" \
    --end "${END}"
