from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.classes.settings import EVENTS
from src.services.scoring import ThresholdTuner, apply_thresholds, score_predictions

TA_EVALUATE = Path(os.environ.get("BIG_DATA_NYCU_STORAGE", "/nonexistent")) / "data/hw2/assignment2/evaluation/evaluate.py"


def random_case(*, seed: int, users: int = 300) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    truth = (generator.random((users, len(EVENTS))) < [0.1, 0.05, 0.1, 0.1, 0.05, 0.5]).astype("int8")
    predictions = (generator.random((users, len(EVENTS))) < 0.3).astype("int8")
    return predictions, truth


def test_hand_computed_score() -> None:
    truth = np.array([[1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]])
    predictions = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1]])
    report = score_predictions(predictions=predictions, truth=truth)
    assert report.per_event_f1 == pytest.approx((2 / 3, 1.0, 1.0, 1.0, 1.0, 2 / 3))
    assert report.micro_f1 == pytest.approx(4 / 6)
    assert report.score == pytest.approx((report.macro_f1 + report.micro_f1) / 2)


@pytest.mark.skipif(not TA_EVALUATE.is_file(), reason="TA evaluate.py not downloaded")
def test_matches_ta_evaluator() -> None:
    spec = importlib.util.spec_from_file_location("ta_evaluate", TA_EVALUATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predictions, truth = random_case(seed=0)
    users = np.arange(len(truth))
    pred_frame = pd.DataFrame(predictions, columns=list(EVENTS)).assign(UserID=users)
    truth_frame = pd.DataFrame(truth, columns=list(EVENTS)).assign(UserID=users)
    expected = module.scores(pred_frame[module.OUTPUT_COLUMNS], truth_frame[module.OUTPUT_COLUMNS])
    report = score_predictions(predictions=predictions, truth=truth)
    assert report.macro_f1 == pytest.approx(expected["macro_f1"])
    assert report.micro_f1 == pytest.approx(expected["micro_f1"])
    assert report.score == pytest.approx(expected["score"])


def test_tuner_not_worse_than_half() -> None:
    generator = np.random.default_rng(1)
    _, truth = random_case(seed=1, users=500)
    probabilities = np.clip(0.35 * truth + 0.6 * generator.random(truth.shape), 0, 1)
    tuner = ThresholdTuner(grid=[round(0.05 * step, 2) for step in range(1, 20)], passes=3)
    thresholds, report = tuner.tune(probabilities=probabilities, truth=truth)
    baseline = score_predictions(
        predictions=apply_thresholds(probabilities=probabilities, thresholds=[0.5] * len(EVENTS)),
        truth=truth,
    )
    tuned = score_predictions(
        predictions=apply_thresholds(probabilities=probabilities, thresholds=thresholds),
        truth=truth,
    )
    assert tuned.score == pytest.approx(report.score)
    assert report.score >= baseline.score
