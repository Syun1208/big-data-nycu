from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.classes.samples import CALIBRATION_SPLIT, HOLDOUT_SPLIT, TRAIN_SPLIT, CutoffSample
from src.data.classes.settings import EVENTS, ExperimentSettings
from src.interface.activity_model import ActivityModel
from src.services.config_loader import settings_to_dict
from src.services.dataset_builder import DatasetBuilder
from src.services.preview import preview_samples
from src.services.sampling import SamplePlanner
from src.services.scoring import ScoreReport, ThresholdTuner, apply_thresholds, score_predictions

logger = logging.getLogger(__name__)

BASELINE_WINDOWS = (7, 28)
THRESHOLDS_FILE = "thresholds.json"
METRICS_FILE = "metrics.json"
IMPORTANCE_FILE = "feature_importance.csv"
HOLDOUT_FILE = "holdout_predictions.parquet"
CALIBRATION_FILE = "calibration_predictions.parquet"
CONFIG_FILE = "config.json"


@dataclass(frozen=True, slots=True)
class RunPaths:
    output_dir: Path
    model_dir: Path

    def prepare(self) -> None:
        for folder in (self.output_dir, self.model_dir):
            folder.mkdir(
                parents=True,
                exist_ok=True,
            )


def write_json(
    *,
    path: Path,
    payload: object,
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def read_thresholds(*, model_dir: Path) -> list[float]:
    values = json.loads((model_dir / THRESHOLDS_FILE).read_text(encoding="utf-8"))
    return [float(values[event]) for event in EVENTS]


def samples_in(
    *,
    samples: Sequence[CutoffSample],
    split: str,
) -> list[CutoffSample]:
    chosen = [sample for sample in samples if sample.cutoff.split == split]
    if not chosen:
        raise ValueError(f"No cutoff in split {split!r}")
    return chosen


def stacked_labels(*, samples: Sequence[CutoffSample]) -> pd.DataFrame:
    return pd.concat([sample.dataset.labels for sample in samples], ignore_index=True)


def stacked_features(*, samples: Sequence[CutoffSample]) -> pd.DataFrame:
    return pd.concat([sample.dataset.features for sample in samples], ignore_index=True)


def activity_baseline(
    *,
    features: pd.DataFrame,
    window_days: int,
) -> np.ndarray:
    columns = [f"count_{event}_{window_days}d" for event in EVENTS]
    return (features[columns].to_numpy() > 0).astype("int8")


def write_split_predictions(
    *,
    path: Path,
    samples: Sequence[CutoffSample],
    probabilities: np.ndarray,
) -> None:
    predictions = pd.concat([sample.dataset.meta for sample in samples], ignore_index=True)
    labels = stacked_labels(samples=samples)
    for index, event in enumerate(EVENTS):
        predictions[f"proba_{event}"] = probabilities[:, index]
        predictions[f"label_{event}"] = labels[event].to_numpy()
    predictions.to_parquet(
        path,
        index=False,
    )


class ExperimentTrainer:
    def __init__(
        self,
        *,
        settings: ExperimentSettings,
        planner: SamplePlanner,
        dataset_builder: DatasetBuilder,
        model: ActivityModel,
        tuner: ThresholdTuner,
        paths: RunPaths,
    ) -> None:
        self._settings = settings
        self._planner = planner
        self._builder = dataset_builder
        self._model = model
        self._tuner = tuner
        self._paths = paths

    def run(self, *, data_end: pd.Timestamp) -> dict:
        self._paths.prepare()
        cutoffs = self._planner.plan_training_cutoffs(data_end=data_end)
        samples = self._builder.build_many(cutoffs=cutoffs)
        train = samples_in(samples=samples, split=TRAIN_SPLIT)
        calibration = samples_in(samples=samples, split=CALIBRATION_SPLIT)
        holdout = samples_in(samples=samples, split=HOLDOUT_SPLIT)
        logger.info(
            "🚀 %s: train %d rows, calibration %d, holdout %d",
            self._model.name,
            sum(len(sample.dataset.meta) for sample in train),
            sum(len(sample.dataset.meta) for sample in calibration),
            sum(len(sample.dataset.meta) for sample in holdout),
        )
        preview_samples(
            title="Train data",
            samples=train,
        )
        started = time.time()
        self._model.fit(
            train=train,
            calibration=calibration,
        )
        train_minutes = (time.time() - started) / 60
        self._model.save(model_dir=self._paths.model_dir)

        calibration_probability = self._model.predict_proba(samples=calibration)
        holdout_probability = self._model.predict_proba(samples=holdout)
        thresholds, calibration_report = self._tuner.tune(
            probabilities=calibration_probability,
            truth=stacked_labels(samples=calibration).to_numpy(),
        )
        holdout_truth = stacked_labels(samples=holdout)
        holdout_report = score_predictions(
            predictions=apply_thresholds(
                probabilities=holdout_probability,
                thresholds=thresholds,
            ),
            truth=holdout_truth.to_numpy(),
        )
        metrics = self._collect_metrics(
            cutoffs=cutoffs,
            data_end=data_end,
            thresholds=thresholds,
            calibration_report=calibration_report,
            holdout_report=holdout_report,
            holdout=holdout,
            train_minutes=train_minutes,
        )
        self._write_outputs(
            thresholds=thresholds,
            metrics=metrics,
            calibration=calibration,
            calibration_probability=calibration_probability,
            holdout=holdout,
            holdout_probability=holdout_probability,
        )
        return metrics

    def _collect_metrics(
        self,
        *,
        cutoffs: Sequence,
        data_end: pd.Timestamp,
        thresholds: list[float],
        calibration_report: ScoreReport,
        holdout_report: ScoreReport,
        holdout: Sequence[CutoffSample],
        train_minutes: float,
    ) -> dict:
        truth = stacked_labels(samples=holdout)
        features = stacked_features(samples=holdout)
        baselines = {
            f"last_{window}d": score_predictions(
                predictions=activity_baseline(
                    features=features,
                    window_days=window,
                ),
                truth=truth.to_numpy(),
            ).as_dict(events=EVENTS)
            for window in BASELINE_WINDOWS
        }
        baselines["push_only"] = score_predictions(
            predictions=np.tile(np.array([0, 0, 0, 0, 0, 1], dtype="int8"), (len(truth), 1)),
            truth=truth.to_numpy(),
        ).as_dict(events=EVENTS)
        logger.info(
            "📈 %s calibration score %.4f (macro %.4f, micro %.4f)",
            self._model.name,
            calibration_report.score,
            calibration_report.macro_f1,
            calibration_report.micro_f1,
        )
        logger.info(
            "📈 %s holdout score %.4f (macro %.4f, micro %.4f). Baselines: %s",
            self._model.name,
            holdout_report.score,
            holdout_report.macro_f1,
            holdout_report.micro_f1,
            ", ".join(f"{name} {values['score']:.4f}" for name, values in baselines.items()),
        )
        for event, threshold, f1 in zip(EVENTS, thresholds, holdout_report.per_event_f1):
            logger.info("   %-18s threshold %.2f, holdout F1 %.4f", event, threshold, f1)
        return {
            "experiment": self._settings.name,
            "model": self._model.name,
            "data_end": str(data_end),
            "cutoffs": {cutoff.label: cutoff.split for cutoff in cutoffs},
            "train_minutes": train_minutes,
            "thresholds": dict(zip(EVENTS, thresholds)),
            "calibration": calibration_report.as_dict(events=EVENTS),
            "holdout": holdout_report.as_dict(events=EVENTS),
            "holdout_baselines": baselines,
            "holdout_positive_rate": truth.mean().to_dict(),
            "holdout_users": len(truth),
        }

    def _write_outputs(
        self,
        *,
        thresholds: list[float],
        metrics: dict,
        calibration: Sequence[CutoffSample],
        calibration_probability: np.ndarray,
        holdout: Sequence[CutoffSample],
        holdout_probability: np.ndarray,
    ) -> None:
        output_dir = self._paths.output_dir
        config = settings_to_dict(settings=self._settings)
        write_json(path=self._paths.model_dir / THRESHOLDS_FILE, payload=dict(zip(EVENTS, thresholds)))
        write_json(path=self._paths.model_dir / CONFIG_FILE, payload=config)
        write_json(path=output_dir / CONFIG_FILE, payload=config)
        write_json(path=output_dir / METRICS_FILE, payload=metrics)
        importance = self._model.feature_importance()
        if importance is not None:
            importance.assign(mean_gain=importance.mean(axis=1)).sort_values("mean_gain", ascending=False).to_csv(
                output_dir / IMPORTANCE_FILE,
                index_label="feature",
            )
        for samples, probabilities, name in (
            (calibration, calibration_probability, CALIBRATION_FILE),
            (holdout, holdout_probability, HOLDOUT_FILE),
        ):
            write_split_predictions(
                path=output_dir / name,
                samples=samples,
                probabilities=probabilities,
            )
        logger.info("💾 Metrics in %s, model in %s", output_dir, self._paths.model_dir)
