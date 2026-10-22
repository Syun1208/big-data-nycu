from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import pytest

from src.data.classes.settings import EVENTS
from src.services.evaluation import SAMPLE_OUTPUT, SAMPLE_TRUTH, SubmissionEvaluator, log_report, run_official_evaluator

EVALUATION_DIR = Path(os.environ.get("BIG_DATA_NYCU_STORAGE", "/nonexistent")) / "data/hw2/assignment2/evaluation"
needs_ta_data = pytest.mark.skipif(not (EVALUATION_DIR / "evaluate.py").is_file(), reason="TA evaluate.py not downloaded")


@needs_ta_data
def test_sample_output_score_matches_evaluate_py(caplog) -> None:
    caplog.set_level(logging.INFO)
    evaluator = SubmissionEvaluator(evaluation_dir=EVALUATION_DIR)
    report = evaluator.evaluate(
        prediction=EVALUATION_DIR / SAMPLE_OUTPUT,
        truth=EVALUATION_DIR / SAMPLE_TRUTH,
    )
    official = run_official_evaluator(
        evaluator=evaluator.evaluator,
        prediction=EVALUATION_DIR / SAMPLE_OUTPUT,
        truth=EVALUATION_DIR / SAMPLE_TRUTH,
        test_input=None,
    )
    assert official.returncode == 0
    assert report.score == pytest.approx(official.score_line(name="Score"), abs=1e-4)
    assert report.macro_f1 == pytest.approx(official.score_line(name="macro-F1"), abs=1e-4)
    assert report.micro_f1 == pytest.approx(official.score_line(name="micro-F1"), abs=1e-4)


@needs_ta_data
def test_perfect_prediction_scores_one(tmp_path) -> None:
    truth = pd.read_csv(EVALUATION_DIR / SAMPLE_TRUTH)
    prediction = tmp_path / "perfect.csv"
    truth.to_csv(prediction, index=False)
    report = SubmissionEvaluator(evaluation_dir=EVALUATION_DIR).evaluate(
        prediction=prediction,
        truth=EVALUATION_DIR / SAMPLE_TRUTH,
    )
    log_report(title="perfect prediction", report=report)
    assert report.score == pytest.approx(1.0)


@needs_ta_data
def test_bad_format_is_rejected(tmp_path, caplog) -> None:
    truth = pd.read_csv(EVALUATION_DIR / SAMPLE_TRUTH)
    prediction = tmp_path / "bad.csv"
    truth.assign(push=2).to_csv(prediction, index=False)
    caplog.set_level(logging.CRITICAL, logger="src.services.evaluation")
    with pytest.raises(RuntimeError, match="rejected"):
        SubmissionEvaluator(evaluation_dir=EVALUATION_DIR).evaluate(
            prediction=prediction,
            truth=EVALUATION_DIR / SAMPLE_TRUTH,
        )
    logging.getLogger(__name__).info("✅ A file with push=2 was rejected by evaluate.py as expected")
    assert list(truth.columns[1:]) == list(EVENTS)
