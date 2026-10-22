#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=python
CONFIG=configs/graphs/render.yaml
OUTPUT_DIR=images

"${PYTHON}" main.py graphs \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
