from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from src.data.classes.settings import EVENTS
from src.services.evaluation import OfficialResult, SubmissionEvaluator, run_official_evaluator
from src.services.scoring import ScoreReport

SCORE_TOLERANCE = 1e-4
FLIPPED_USERS = 500


@pytest.fixture(scope="module")
def submission_evaluator(evaluator: Path, test_input: Path) -> SubmissionEvaluator:
    return SubmissionEvaluator(evaluator=evaluator, test_input=test_input)


@pytest.fixture(scope="module")
def truth_file(sample_submission: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    truth = pd.read_csv(sample_submission)
    columns = list(EVENTS)
    truth.loc[: FLIPPED_USERS - 1, columns] = 1 - truth.loc[: FLIPPED_USERS - 1, columns]
    path = tmp_path_factory.mktemp("truth") / "truth.csv"
    truth.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def sample_report(submission_evaluator: SubmissionEvaluator, sample_submission: Path, truth_file: Path) -> ScoreReport:
    return submission_evaluator.evaluate(
        prediction=sample_submission,
        truth=truth_file,
    )


@pytest.fixture(scope="module")
def official_result(evaluator: Path, sample_submission: Path, truth_file: Path) -> OfficialResult:
    return run_official_evaluator(
        evaluator=evaluator,
        prediction=sample_submission,
        truth=truth_file,
        test_input=None,
    )


def test_official_evaluator_accepts_the_sample_submission(official_result: OfficialResult) -> None:
    assert official_result.returncode == 0


def test_sample_submission_lists_every_test_user(evaluator: Path, sample_submission: Path, test_input: Path) -> None:
    result = run_official_evaluator(
        evaluator=evaluator,
        prediction=sample_submission,
        truth=None,
        test_input=test_input,
    )
    assert result.returncode == 0


@pytest.mark.parametrize(
    ("metric", "official_name"),
    [
        pytest.param("score", "Score", id="score"),
        pytest.param("macro_f1", "macro-F1", id="macro_f1"),
        pytest.param("micro_f1", "micro-F1", id="micro_f1"),
    ],
)
def test_metric_matches_evaluate_py(
    sample_report: ScoreReport,
    official_result: OfficialResult,
    metric: str,
    official_name: str,
) -> None:
    assert getattr(sample_report, metric) == pytest.approx(official_result.score_line(name=official_name), abs=SCORE_TOLERANCE)


def test_perfect_prediction_scores_one(submission_evaluator: SubmissionEvaluator, truth_file: Path) -> None:
    report = submission_evaluator.evaluate(
        prediction=truth_file,
        truth=truth_file,
    )
    assert report.score == pytest.approx(1.0)


def test_non_binary_value_is_rejected(
    submission_evaluator: SubmissionEvaluator,
    truth_file: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    prediction = tmp_path / "bad.csv"
    pd.read_csv(truth_file).assign(push=2).to_csv(prediction, index=False)
    caplog.set_level(logging.CRITICAL, logger="src.services.evaluation")
    with pytest.raises(RuntimeError, match="rejected"):
        submission_evaluator.evaluate(
            prediction=prediction,
            truth=truth_file,
        )
