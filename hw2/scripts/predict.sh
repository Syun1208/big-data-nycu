#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=python
GPU_IDS=0
ENSEMBLE_CONFIGS=(
    configs/xgboost/exp6.long.short_windows.yaml
    configs/xgboost/exp7.long.short_windows.seed7.yaml
    configs/rdl_gnn/exp5.hybrid.short_windows.seed1.yaml
    configs/rdl_gnn/exp5.hybrid.short_windows.seed42.yaml
    configs/rdl_gnn/exp6.gat.hybrid.short_windows.seed1.yaml
    configs/rdl_gnn/exp6.gat.hybrid.short_windows.seed42.yaml
    configs/rdl_gnn/exp7.transformer.hybrid.short_windows.seed1.yaml
    configs/rdl_gnn/exp7.transformer.hybrid.short_windows.seed42.yaml
)
CONFIGS=(
    configs/xgboost/exp1.yaml
    configs/rdl_gnn/exp1.yaml
    "${ENSEMBLE_CONFIGS[@]}"
)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS="," read -r -a GPUS <<< "${GPU_IDS}"
export CUDA_VISIBLE_DEVICES=${GPUS[0]}

for CONFIG in "${CONFIGS[@]}"; do
    "${PYTHON}" main.py predict \
        --config "${CONFIG}" \
        "$@"
done

"${PYTHON}" main.py ensemble \
    --name ensemble.short_windows \
    --step 0.2 \
    --configs "${ENSEMBLE_CONFIGS[@]}"

"${PYTHON}" main.py compare \
    --configs "${CONFIGS[@]}"
