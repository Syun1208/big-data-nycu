from __future__ import annotations

import gc
import time
from collections.abc import Callable
from typing import TypeVar

ResultT = TypeVar("ResultT")
BYTES_PER_GB = 1024**3


def release_cuda_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def time_cuda_call(*, action: Callable[[], ResultT]) -> tuple[ResultT, float]:
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    result = action()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def read_free_memory_gb(*, device_index: int) -> float:
    import torch

    free_bytes, _ = torch.cuda.mem_get_info(device_index)
    return free_bytes / BYTES_PER_GB
