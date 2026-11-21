#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID=0
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
    configs/sthn/exp2.hybrid.short_windows.yaml
    configs/stgnn/exp2.hybrid.short_windows.yaml
    "${ENSEMBLE_CONFIGS[@]}"
)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU_ID}

for CONFIG in "${CONFIGS[@]}"; do
    python main.py train \
        --config "${CONFIG}"
done

python main.py ensemble \
    --name ensemble.short_windows \
    --step 0.2 \
    --configs "${ENSEMBLE_CONFIGS[@]}"

python main.py compare \
    --configs "${CONFIGS[@]}"
