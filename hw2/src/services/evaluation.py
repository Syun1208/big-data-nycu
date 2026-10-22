from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.classes.settings import EVENTS
from src.services.scoring import ScoreReport, score_predictions

logger = logging.getLogger(__name__)

USER_COLUMN = "UserID"
SAMPLE_TRUTH = Path("examples") / "sample_truth.csv"
SAMPLE_OUTPUT = Path("examples") / "sample_output.csv"


@dataclass(frozen=True, slots=True)
class OfficialResult:
    returncode: int
    output: str

    def score_line(self, *, name: str) -> float | None:
        for line in self.output.splitlines():
            if line.startswith(f"{name}:"):
                return float(line.split(":", 1)[1].strip().rstrip("%")) / 100
        return None


def run_official_evaluator(
    *,
    evaluator: Path,
    prediction: Path,
    truth: Path | None,
    test_input: Path | None,
) -> OfficialResult:
    command = [sys.executable, str(evaluator), "--pred", str(prediction)]
    if truth is not None:
        command.extend(["--truth", str(truth)])
    elif test_input is not None:
        command.extend(["--input", str(test_input)])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return OfficialResult(
        returncode=result.returncode,
        output=(result.stdout + result.stderr).strip(),
    )


def score_files(
    *,
    prediction: Path,
    truth: Path,
) -> ScoreReport:
    predicted = pd.read_csv(prediction).set_index(USER_COLUMN)
    expected = pd.read_csv(truth).set_index(USER_COLUMN)
    missing = expected.index.difference(predicted.index)
    if len(missing):
        raise ValueError(f"{len(missing)} truth users have no prediction in {prediction}")
    columns = list(EVENTS)
    return score_predictions(
        predictions=predicted.loc[expected.index, columns].to_numpy(),
        truth=expected[columns].to_numpy(),
    )


def log_report(
    *,
    title: str,
    report: ScoreReport,
) -> None:
    logger.info(
        "📈 %s: score %.2f%%, macro-F1 %.2f%%, micro-F1 %.2f%%",
        title,
        100 * report.score,
        100 * report.macro_f1,
        100 * report.micro_f1,
    )
    for event, f1 in zip(EVENTS, report.per_event_f1):
        logger.info("   %-18s F1 %.2f%%", event, 100 * f1)


class SubmissionEvaluator:
    def __init__(self, *, evaluation_dir: Path) -> None:
        self._evaluation_dir = evaluation_dir

    @property
    def evaluator(self) -> Path:
        return self._evaluation_dir / "evaluate.py"

    def evaluate(
        self,
        *,
        prediction: Path,
        truth: Path | None,
    ) -> ScoreReport | None:
        test_input = self._evaluation_dir / "test_input.csv"
        official = run_official_evaluator(
            evaluator=self.evaluator,
            prediction=prediction,
            truth=truth,
            test_input=test_input if test_input.is_file() else None,
        )
        for line in official.output.splitlines():
            logger.info("evaluate.py | %s", line)
        if official.returncode != 0:
            raise RuntimeError(f"evaluate.py rejected {prediction}")
        if truth is None:
            return None
        report = score_files(
            prediction=prediction,
            truth=truth,
        )
        log_report(
            title=f"{prediction.name} vs {truth.name}",
            report=report,
        )
        official_score = official.score_line(name="Score")
        if official_score is not None and abs(official_score - report.score) > 1e-4:
            raise RuntimeError(f"Score mismatch: evaluate.py {official_score:.4f}, ours {report.score:.4f}")
        return report
