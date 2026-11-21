#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_IDS=6
STORAGE=/mnt/HDD1/longpm/big-data-nycu
PRED=${STORAGE}/outputs/hw2/submission/output_314540080.csv
TRUTH=
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

CUDA_VISIBLE_DEVICES= python -m pytest \
    "$@"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS="," read -r -a GPUS <<< "${GPU_IDS}"
export CUDA_VISIBLE_DEVICES=${GPUS[0]}

for CONFIG in "${CONFIGS[@]}"; do
    python main.py predict \
        --config "${CONFIG}"
done

python main.py ensemble \
    --name ensemble.short_windows \
    --step 0.2 \
    --configs "${ENSEMBLE_CONFIGS[@]}"

python main.py compare \
    --configs "${CONFIGS[@]}"

python main.py diagnose \
    --configs "${CONFIGS[@]}"

if [[ -n "${TRUTH}" ]]; then
    python main.py evaluate \
        --pred "${PRED}" \
        --truth "${TRUTH}"
else
    python main.py evaluate \
        --pred "${PRED}" \
        --format-only
fi

python main.py benchmark
