#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=python
CONFIG=configs/xgboost/exp1.yaml

"${PYTHON}" main.py download-data \
    --config "${CONFIG}" \
    --keep-snapshot \
    "$@"
