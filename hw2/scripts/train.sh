#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=python
GPU_IDS=0,5
MIN_FREE_GPU_GB=4
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

"${PYTHON}" main.py build-features \
    --config configs/xgboost/exp6.long.short_windows.yaml

"${PYTHON}" main.py build-features \
    --config configs/xgboost/exp1.yaml

PIDS=()
stop_jobs() {
    if (( ${#PIDS[@]} > 0 )); then
        kill "${PIDS[@]}" 2> /dev/null || true
        wait "${PIDS[@]}" 2> /dev/null || true
    fi
}
trap stop_jobs EXIT
trap "exit 130" INT TERM

for INDEX in "${!CONFIGS[@]}"; do
    GPU=${GPUS[$((INDEX % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" main.py train \
        --config "${CONFIGS[${INDEX}]}" \
        --min-free-gpu-gb "${MIN_FREE_GPU_GB}" \
        "$@" &
    PIDS+=($!)
    if (( ${#PIDS[@]} >= ${#GPUS[@]} )); then
        wait "${PIDS[0]}"
        PIDS=("${PIDS[@]:1}")
    fi
done
while (( ${#PIDS[@]} > 0 )); do
    wait "${PIDS[0]}"
    PIDS=("${PIDS[@]:1}")
done
trap - EXIT

"${PYTHON}" main.py ensemble \
    --name ensemble.short_windows \
    --step 0.2 \
    --configs "${ENSEMBLE_CONFIGS[@]}"

"${PYTHON}" main.py compare \
    --configs "${CONFIGS[@]}"
