#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG=configs/graphs/render.yaml
OUTPUT_DIR=images

python main.py graphs \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
