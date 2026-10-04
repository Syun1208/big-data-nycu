from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FLATTENED_EVALUATION_KEYS = ("accuracy", "correct", "total", "unparsed", "eval_seconds")


@dataclass(frozen=True, slots=True)
class PiSSAInitStats:
    name: str
    rank: int
    component: str
    relative_error_stored: float


@dataclass(frozen=True, slots=True)
class AnswerCheck:
    pred_answer: str | None
    gold_answer: str
    correct: bool
    parsed: bool
    extraction_source: str | None


@dataclass(frozen=True, slots=True)
class LayerSpectrum:
    layer_index: int
    module_name: str
    weight_shape: tuple[int, int]
    singular_values: tuple[float, ...]
    cumulative_energy_percent: tuple[float, ...]
    energy_threshold_percent: float
    minimum_rank: int


@dataclass(frozen=True, slots=True)
class TaskEvaluation:
    task: str
    accuracy: float
    correct: int
    total: int
    unparsed: int
    eval_seconds: float
    evaluation_protocol: str
    records: list[dict[str, Any]] = field(default_factory=list)
