from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScoreReport:
    macro_f1: float
    micro_f1: float
    score: float
    per_event_f1: tuple[float, ...]

    def as_dict(self, *, events: Sequence[str]) -> dict:
        return {
            "macro_f1": self.macro_f1,
            "micro_f1": self.micro_f1,
            "score": self.score,
            "per_event_f1": dict(zip(events, self.per_event_f1)),
        }


def f1_from_counts(
    *,
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
) -> np.ndarray:
    tp, fp, fn = (np.asarray(value, dtype="float64") for value in (tp, fp, fn))
    denominator = 2 * tp + fp + fn
    return np.where(denominator > 0, 2 * tp / np.where(denominator > 0, denominator, 1.0), 1.0)


def confusion_counts(
    *,
    predictions: np.ndarray,
    truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = predictions == 1
    actual = truth == 1
    return (
        (predicted & actual).sum(axis=0),
        (predicted & ~actual).sum(axis=0),
        (~predicted & actual).sum(axis=0),
    )


def score_counts(
    *,
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
) -> ScoreReport:
    per_event = f1_from_counts(
        tp=tp,
        fp=fp,
        fn=fn,
    )
    macro = float(per_event.mean())
    micro = float(
        f1_from_counts(
            tp=np.sum(tp),
            fp=np.sum(fp),
            fn=np.sum(fn),
        )
    )
    return ScoreReport(
        macro_f1=macro,
        micro_f1=micro,
        score=(macro + micro) / 2,
        per_event_f1=tuple(float(value) for value in per_event),
    )


def score_predictions(
    *,
    predictions: np.ndarray,
    truth: np.ndarray,
) -> ScoreReport:
    tp, fp, fn = confusion_counts(
        predictions=predictions,
        truth=truth,
    )
    return score_counts(
        tp=tp,
        fp=fp,
        fn=fn,
    )


def apply_thresholds(
    *,
    probabilities: np.ndarray,
    thresholds: Sequence[float],
) -> np.ndarray:
    return (probabilities >= np.asarray(thresholds)[None, :]).astype("int8")


class ThresholdTuner:
    def __init__(
        self,
        *,
        grid: Sequence[float],
        passes: int,
    ) -> None:
        self._grid = np.asarray(grid, dtype="float64")
        self._passes = passes

    def tune(
        self,
        *,
        probabilities: np.ndarray,
        truth: np.ndarray,
    ) -> tuple[list[float], ScoreReport]:
        n_events = probabilities.shape[1]
        actual = truth == 1
        grid_counts = []
        for column in range(n_events):
            predicted = probabilities[:, column][:, None] >= self._grid[None, :]
            positive = actual[:, column][:, None]
            grid_counts.append(
                (
                    (predicted & positive).sum(axis=0),
                    (predicted & ~positive).sum(axis=0),
                    (~predicted & positive).sum(axis=0),
                )
            )
        choice = [int(np.argmin(np.abs(self._grid - 0.5)))] * n_events
        best = self._score_choice(
            grid_counts=grid_counts,
            choice=choice,
        )
        for step in range(self._passes):
            improved = False
            for column in range(n_events):
                for index in range(len(self._grid)):
                    if index == choice[column]:
                        continue
                    candidate = list(choice)
                    candidate[column] = index
                    report = self._score_choice(
                        grid_counts=grid_counts,
                        choice=candidate,
                    )
                    if report.score > best.score + 1e-12:
                        best, choice, improved = report, candidate, True
            logger.debug("Threshold pass %d: score %.4f", step + 1, best.score)
            if not improved:
                break
        return [float(self._grid[index]) for index in choice], best

    @staticmethod
    def _score_choice(
        *,
        grid_counts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        choice: Sequence[int],
    ) -> ScoreReport:
        tp = np.array([counts[0][index] for counts, index in zip(grid_counts, choice)])
        fp = np.array([counts[1][index] for counts, index in zip(grid_counts, choice)])
        fn = np.array([counts[2][index] for counts, index in zip(grid_counts, choice)])
        return score_counts(
            tp=tp,
            fp=fp,
            fn=fn,
        )
