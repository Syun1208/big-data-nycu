from __future__ import annotations

from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src.data.classes.settings import EVENTS
from src.services.scoring import ScoreReport, ThresholdTuner, apply_thresholds, score_predictions

POSITIVE_RATES = [0.1, 0.05, 0.1, 0.1, 0.05, 0.5]
PREDICTED_RATE = 0.3
RANDOM_USERS = 300
TUNING_USERS = 500
DEFAULT_THRESHOLD = 0.5
THRESHOLD_GRID = tuple(round(0.05 * step, 2) for step in range(1, 20))
METRICS = ["macro_f1", "micro_f1", "score"]


@pytest.fixture
def hand_truth() -> np.ndarray:
    return np.array(
        [
            [1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
        ]
    )


@pytest.fixture
def hand_predictions() -> np.ndarray:
    return np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1],
        ]
    )


@pytest.fixture
def hand_report(hand_predictions: np.ndarray, hand_truth: np.ndarray) -> ScoreReport:
    return score_predictions(predictions=hand_predictions, truth=hand_truth)


@pytest.fixture
def random_truth() -> np.ndarray:
    generator = np.random.default_rng(0)
    return (generator.random((RANDOM_USERS, len(EVENTS))) < POSITIVE_RATES).astype("int8")


@pytest.fixture
def random_predictions() -> np.ndarray:
    generator = np.random.default_rng(1)
    return (generator.random((RANDOM_USERS, len(EVENTS))) < PREDICTED_RATE).astype("int8")


@pytest.fixture
def tuning_truth() -> np.ndarray:
    generator = np.random.default_rng(2)
    return (generator.random((TUNING_USERS, len(EVENTS))) < POSITIVE_RATES).astype("int8")


@pytest.fixture
def tuning_probabilities(tuning_truth: np.ndarray) -> np.ndarray:
    generator = np.random.default_rng(3)
    return np.clip(0.35 * tuning_truth + 0.6 * generator.random(tuning_truth.shape), 0, 1)


@pytest.fixture
def tuned(tuning_probabilities: np.ndarray, tuning_truth: np.ndarray) -> tuple[list[float], ScoreReport]:
    tuner = ThresholdTuner(grid=THRESHOLD_GRID, passes=3)
    return tuner.tune(probabilities=tuning_probabilities, truth=tuning_truth)


def test_hand_computed_per_event_f1(hand_report: ScoreReport) -> None:
    assert hand_report.per_event_f1 == pytest.approx((2 / 3, 1.0, 1.0, 1.0, 1.0, 2 / 3))


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        pytest.param("macro_f1", 8 / 9, id="macro_f1"),
        pytest.param("micro_f1", 2 / 3, id="micro_f1"),
        pytest.param("score", 7 / 9, id="score"),
    ],
)
def test_hand_computed_metric(hand_report: ScoreReport, metric: str, expected: float) -> None:
    assert getattr(hand_report, metric) == pytest.approx(expected)


@pytest.mark.parametrize("metric", METRICS)
def test_matches_ta_evaluator(
    ta_evaluate: ModuleType,
    random_predictions: np.ndarray,
    random_truth: np.ndarray,
    metric: str,
) -> None:
    users = np.arange(len(random_truth))
    predicted = pd.DataFrame(random_predictions, columns=list(EVENTS)).assign(UserID=users)
    actual = pd.DataFrame(random_truth, columns=list(EVENTS)).assign(UserID=users)
    expected = ta_evaluate.scores(predicted[ta_evaluate.OUTPUT_COLUMNS], actual[ta_evaluate.OUTPUT_COLUMNS])
    report = score_predictions(predictions=random_predictions, truth=random_truth)
    assert getattr(report, metric) == pytest.approx(expected[metric])


def test_tuner_report_matches_its_thresholds(
    tuned: tuple[list[float], ScoreReport],
    tuning_probabilities: np.ndarray,
    tuning_truth: np.ndarray,
) -> None:
    thresholds, report = tuned
    applied = score_predictions(
        predictions=apply_thresholds(probabilities=tuning_probabilities, thresholds=thresholds),
        truth=tuning_truth,
    )
    assert applied.score == pytest.approx(report.score)


def test_tuned_thresholds_beat_the_default_threshold(
    tuned: tuple[list[float], ScoreReport],
    tuning_probabilities: np.ndarray,
    tuning_truth: np.ndarray,
) -> None:
    _, report = tuned
    default = score_predictions(
        predictions=apply_thresholds(probabilities=tuning_probabilities, thresholds=[DEFAULT_THRESHOLD] * len(EVENTS)),
        truth=tuning_truth,
    )
    assert report.score >= default.score
