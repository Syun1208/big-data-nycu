from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ORIGINAL_NOTEBOOK = ROOT / "src" / "notebooks" / "finetune_q1-q3.ipynb"
ANSWER_PARSING_MARKERS = ("def strip_pissa_math_string",)
PISSA_MARKERS = ("class PiSSALinear", "def pissa_from_linear_exact", "def inject_pissa")


def find_code_cell(
    *,
    cells: list[dict[str, Any]],
    marker: str,
) -> str:
    matches = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code" and marker in "".join(cell["source"])]
    if len(matches) != 1:
        raise LookupError(f"expected exactly one code cell containing {marker!r}, found {len(matches)}")
    return matches[0]


def load_original_namespace(*, markers: tuple[str, ...]) -> dict[str, Any]:
    import gc
    import math
    import re
    import time
    from dataclasses import dataclass
    from fractions import Fraction

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cells = json.loads(ORIGINAL_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    namespace: dict[str, Any] = {
        "gc": gc,
        "math": math,
        "re": re,
        "time": time,
        "dataclass": dataclass,
        "Fraction": Fraction,
        "torch": torch,
        "nn": nn,
        "F": F,
        "SVD_DEVICE": torch.device("cpu"),
    }
    for marker in markers:
        exec(
            find_code_cell(
                cells=cells,
                marker=marker,
            ),
            namespace,
        )
    return namespace


@pytest.fixture(scope="session")
def original_answer_parsing() -> dict[str, Any]:
    return load_original_namespace(markers=ANSWER_PARSING_MARKERS)


@pytest.fixture(scope="session")
def original_pissa() -> dict[str, Any]:
    return load_original_namespace(markers=PISSA_MARKERS)
