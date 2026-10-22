from __future__ import annotations

import os
import subprocess

CUDA_VISIBLE_DEVICES_VARIABLE = "CUDA_VISIBLE_DEVICES"
BYTES_PER_MIB = 1024 * 1024


def visible_gpu_ids() -> list[str]:
    value = os.environ.get(CUDA_VISIBLE_DEVICES_VARIABLE, "").strip()
    return [item for item in value.split(",") if item]


def read_free_memory_mib(*, server_gpu_id: str) -> int:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "-i", server_gpu_id,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return int(output.strip().splitlines()[0])
