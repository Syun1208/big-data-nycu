from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from src.services.evaluation import SAMPLE_OUTPUT, SAMPLE_TRUTH, OfficialResult, SubmissionEvaluator, run_official_evaluator
from src.services.scoring import ScoreReport

SCORE_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def sample_report(evaluation_dir: Path) -> ScoreReport:
    return SubmissionEvaluator(evaluation_dir=evaluation_dir).evaluate(
        prediction=evaluation_dir / SAMPLE_OUTPUT,
        truth=evaluation_dir / SAMPLE_TRUTH,
    )


@pytest.fixture(scope="module")
def official_result(evaluation_dir: Path) -> OfficialResult:
    return run_official_evaluator(
        evaluator=evaluation_dir / "evaluate.py",
        prediction=evaluation_dir / SAMPLE_OUTPUT,
        truth=evaluation_dir / SAMPLE_TRUTH,
        test_input=None,
    )


@pytest.fixture
def sample_truth(evaluation_dir: Path) -> pd.DataFrame:
    return pd.read_csv(evaluation_dir / SAMPLE_TRUTH)


def test_official_evaluator_accepts_the_sample_output(official_result: OfficialResult) -> None:
    assert official_result.returncode == 0


@pytest.mark.parametrize(
    ("metric", "official_name"),
    [
        pytest.param("score", "Score", id="score"),
        pytest.param("macro_f1", "macro-F1", id="macro_f1"),
        pytest.param("micro_f1", "micro-F1", id="micro_f1"),
    ],
)
def test_sample_output_metric_matches_evaluate_py(
    sample_report: ScoreReport,
    official_result: OfficialResult,
    metric: str,
    official_name: str,
) -> None:
    assert getattr(sample_report, metric) == pytest.approx(official_result.score_line(name=official_name), abs=SCORE_TOLERANCE)


def test_perfect_prediction_scores_one(evaluation_dir: Path, sample_truth: pd.DataFrame, tmp_path: Path) -> None:
    prediction = tmp_path / "perfect.csv"
    sample_truth.to_csv(prediction, index=False)
    report = SubmissionEvaluator(evaluation_dir=evaluation_dir).evaluate(
        prediction=prediction,
        truth=evaluation_dir / SAMPLE_TRUTH,
    )
    assert report.score == pytest.approx(1.0)


def test_non_binary_value_is_rejected(
    evaluation_dir: Path,
    sample_truth: pd.DataFrame,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    prediction = tmp_path / "bad.csv"
    sample_truth.assign(push=2).to_csv(prediction, index=False)
    caplog.set_level(logging.CRITICAL, logger="src.services.evaluation")
    with pytest.raises(RuntimeError, match="rejected"):
        SubmissionEvaluator(evaluation_dir=evaluation_dir).evaluate(
            prediction=prediction,
            truth=evaluation_dir / SAMPLE_TRUTH,
        )
