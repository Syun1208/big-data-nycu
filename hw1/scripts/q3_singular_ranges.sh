#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=python
RUN_DATE=$(date +%F)
OUTPUT_DIR=outputs/${RUN_DATE}_q3_singular_ranges
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > best {best = $2; selected = $1} END {print selected}')

cd "${PROJECT_ROOT}"
Q1_RESULTS=$(ls -1d outputs/[0-9]*_q1_q2_rank_sweep/benchmark_results.csv 2>/dev/null | tail -n 1 || true)

RUNS=(--run 16:p25 --run 16:p50 --run 16:bottom)
REPORT_INPUTS=("${OUTPUT_DIR}/benchmark_results.csv")
if [[ -n "${Q1_RESULTS}" && -f "${Q1_RESULTS}" ]]; then
    REPORT_INPUTS=("${Q1_RESULTS}" "${REPORT_INPUTS[@]}")
else
    RUNS=(--run 16:default "${RUNS[@]}")
fi

"${PYTHON}" main.py finetune \
    "${RUNS[@]}" \
    --skip-baseline \
    --output-dir "${OUTPUT_DIR}" \
    "$@"

"${PYTHON}" main.py report \
    --results "${REPORT_INPUTS[@]}" \
    --output-dir "${OUTPUT_DIR}/report"
