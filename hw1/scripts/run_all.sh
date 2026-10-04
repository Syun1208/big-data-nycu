#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "${SCRIPT_DIR}/q4_spectrum.sh"
bash "${SCRIPT_DIR}/q1_q2_rank_sweep.sh" "$@"
bash "${SCRIPT_DIR}/q3_singular_ranges.sh" "$@"
