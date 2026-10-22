from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.classes.settings import EVENTS
from src.services.evaluation import USER_COLUMN, SubmissionEvaluator
from src.services.prediction import OUTPUT_FILE, PROBABILITY_FILE
from src.services.scoring import ScoreReport, ThresholdTuner, apply_thresholds, score_predictions
from src.services.training import CALIBRATION_FILE, HOLDOUT_FILE, METRICS_FILE, THRESHOLDS_FILE, write_json

logger = logging.getLogger(__name__)

KEY_COLUMNS = ("github_user_id", "cutoff")
WEIGHTS_FILE = "weights.json"


def simplex_grid(
    *,
    size: int,
    step: float,
) -> list[tuple[float, ...]]:
    units = round(1 / step)
    grid = []
    for parts in itertools.product(range(units + 1), repeat=size):
        if sum(parts) == units:
            grid.append(tuple(part / units for part in parts))
    return grid


def load_split(
    *,
    run_dirs: Sequence[Path],
    name: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    frames = [pd.read_parquet(run_dir / name).set_index(list(KEY_COLUMNS)).sort_index() for run_dir in run_dirs]
    keys = frames[0].index
    for run_dir, frame in zip(run_dirs, frames):
        if not frame.index.equals(keys):
            raise ValueError(f"{run_dir} has different users or cutoffs in {name}. Ensemble members must share the sampling settings.")
    probabilities = np.stack([frame[[f"proba_{event}" for event in EVENTS]].to_numpy() for frame in frames])
    truth = frames[0][[f"label_{event}" for event in EVENTS]].to_numpy()
    return probabilities, truth, frames[0].reset_index()[list(KEY_COLUMNS)]


class EnsembleBuilder:
    def __init__(
        self,
        *,
        tuner: ThresholdTuner,
        step: float,
        evaluation_dir: Path,
    ) -> None:
        self._tuner = tuner
        self._step = step
        self._evaluation_dir = evaluation_dir

    def build(
        self,
        *,
        run_dirs: Sequence[Path],
        output_dir: Path,
    ) -> dict:
        calibration, calibration_truth, _ = load_split(run_dirs=run_dirs, name=CALIBRATION_FILE)
        holdout, holdout_truth, holdout_keys = load_split(run_dirs=run_dirs, name=HOLDOUT_FILE)
        best = None
        for weights in simplex_grid(size=len(run_dirs), step=self._step):
            mixed = np.tensordot(np.asarray(weights), calibration, axes=1)
            thresholds, report = self._tuner.tune(
                probabilities=mixed,
                truth=calibration_truth,
            )
            if best is None or report.score > best[2].score:
                best = (weights, thresholds, report)
        weights, thresholds, calibration_report = best
        holdout_report = score_predictions(
            predictions=apply_thresholds(
                probabilities=np.tensordot(np.asarray(weights), holdout, axes=1),
                thresholds=thresholds,
            ),
            truth=holdout_truth,
        )
        self._log(
            run_dirs=run_dirs,
            weights=weights,
            calibration_report=calibration_report,
            holdout_report=holdout_report,
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        member_metrics = json.loads((run_dirs[0] / METRICS_FILE).read_text(encoding="utf-8"))
        metrics = {
            "experiment": "ensemble",
            "model": "ensemble",
            "members": [run_dir.name for run_dir in run_dirs],
            "weights": dict(zip((run_dir.name for run_dir in run_dirs), weights)),
            "thresholds": dict(zip(EVENTS, thresholds)),
            "calibration": calibration_report.as_dict(events=EVENTS),
            "holdout": holdout_report.as_dict(events=EVENTS),
            "holdout_baselines": member_metrics["holdout_baselines"],
            "train_minutes": None,
        }
        write_json(path=output_dir / METRICS_FILE, payload=metrics)
        write_json(path=output_dir / THRESHOLDS_FILE, payload=dict(zip(EVENTS, thresholds)))
        write_json(path=output_dir / WEIGHTS_FILE, payload=metrics["weights"])
        mixed_holdout = np.tensordot(np.asarray(weights), holdout, axes=1)
        frame = holdout_keys.copy()
        for index, event in enumerate(EVENTS):
            frame[f"proba_{event}"] = mixed_holdout[:, index]
            frame[f"label_{event}"] = holdout_truth[:, index]
        frame.to_parquet(output_dir / HOLDOUT_FILE, index=False)
        self._write_submission(
            run_dirs=run_dirs,
            weights=weights,
            thresholds=thresholds,
            output_dir=output_dir,
        )
        return metrics

    def _log(
        self,
        *,
        run_dirs: Sequence[Path],
        weights: Sequence[float],
        calibration_report: ScoreReport,
        holdout_report: ScoreReport,
    ) -> None:
        for run_dir, weight in zip(run_dirs, weights):
            logger.info("   weight %.2f  %s", weight, run_dir.name)
        logger.info("📈 ensemble calibration score %.4f", calibration_report.score)
        logger.info(
            "📈 ensemble holdout score %.4f (macro %.4f, micro %.4f)",
            holdout_report.score,
            holdout_report.macro_f1,
            holdout_report.micro_f1,
        )
        for event, f1 in zip(EVENTS, holdout_report.per_event_f1):
            logger.info("   %-18s holdout F1 %.4f", event, f1)

    def _write_submission(
        self,
        *,
        run_dirs: Sequence[Path],
        weights: Sequence[float],
        thresholds: Sequence[float],
        output_dir: Path,
    ) -> None:
        if not all((run_dir / PROBABILITY_FILE).is_file() for run_dir in run_dirs):
            logger.warning("Some members have no %s. Run predict first, then ensemble again.", PROBABILITY_FILE)
            return
        frames = [pd.read_csv(run_dir / PROBABILITY_FILE).set_index(USER_COLUMN).sort_index() for run_dir in run_dirs]
        users = frames[0].index
        if not all(frame.index.equals(users) for frame in frames):
            raise ValueError("Members predicted different test users. Predict all members on the same data release.")
        mixed = sum(weight * frame[list(EVENTS)].to_numpy() for weight, frame in zip(weights, frames))
        predictions = pd.DataFrame(
            apply_thresholds(
                probabilities=mixed,
                thresholds=thresholds,
            ),
            columns=list(EVENTS),
            index=users,
        )
        order = pd.read_csv(run_dirs[0] / OUTPUT_FILE)[USER_COLUMN]
        predictions = predictions.reindex(order.to_numpy(), fill_value=0).astype("int64").rename_axis(USER_COLUMN).reset_index()
        output_path = output_dir / OUTPUT_FILE
        predictions.to_csv(
            output_path,
            index=False,
        )
        pd.DataFrame(mixed, columns=list(EVENTS), index=users).reset_index().to_csv(output_dir / PROBABILITY_FILE, index=False)
        logger.info("💾 Wrote %s (%d users)", output_path, len(predictions))
        SubmissionEvaluator(evaluation_dir=self._evaluation_dir).evaluate(
            prediction=output_path,
            truth=None,
        )
