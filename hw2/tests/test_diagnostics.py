from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.classes.settings import EVENTS
from src.services.diagnostics import confusion_table, read_run_thresholds
from src.services.training import METRICS_FILE, THRESHOLDS_FILE, write_json

COUNT_COLUMNS = ["tn", "fp", "fn", "tp"]
SAVED_THRESHOLDS = [0.2, 0.3, 0.25, 0.3, 0.35, 0.4]
METRIC_THRESHOLDS = [0.1] * len(EVENTS)


@pytest.fixture
def event_truth() -> np.ndarray:
    return np.array(
        [
            [1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0],
        ]
    )


@pytest.fixture
def event_predictions() -> np.ndarray:
    return np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1],
        ]
    )


@pytest.fixture
def table(event_truth: np.ndarray, event_predictions: np.ndarray) -> pd.DataFrame:
    return confusion_table(predictions=event_predictions, truth=event_truth)


@pytest.fixture
def run_with_metrics(tmp_path: Path) -> Path:
    write_json(path=tmp_path / METRICS_FILE, payload={"thresholds": dict(zip(EVENTS, METRIC_THRESHOLDS))})
    return tmp_path


@pytest.fixture
def run_with_thresholds(run_with_metrics: Path) -> Path:
    write_json(path=run_with_metrics / THRESHOLDS_FILE, payload=dict(zip(EVENTS, SAVED_THRESHOLDS)))
    return run_with_metrics


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        pytest.param("open_pr", [0, 1, 1, 1], id="open_pr"),
        pytest.param("open_issue", [2, 0, 0, 1], id="open_issue"),
        pytest.param("issue_comment", [3, 0, 0, 0], id="issue_comment"),
        pytest.param("push", [0, 1, 1, 1], id="push"),
    ],
)
def test_confusion_counts(table: pd.DataFrame, event: str, expected: list[int]) -> None:
    assert table.loc[event, COUNT_COLUMNS].tolist() == expected


def test_confusion_counts_cover_every_user(table: pd.DataFrame, event_truth: np.ndarray) -> None:
    assert set(table[COUNT_COLUMNS].sum(axis=1)) == {len(event_truth)}


def test_f1_of_an_event_without_positives_is_one(table: pd.DataFrame) -> None:
    assert table.loc["issue_comment", "f1"] == pytest.approx(1.0)


def test_saved_thresholds_are_preferred(run_with_thresholds: Path) -> None:
    assert read_run_thresholds(run_dir=run_with_thresholds) == pytest.approx(SAVED_THRESHOLDS)


def test_metrics_thresholds_are_the_fallback(run_with_metrics: Path) -> None:
    assert read_run_thresholds(run_dir=run_with_metrics) == pytest.approx(METRIC_THRESHOLDS)
