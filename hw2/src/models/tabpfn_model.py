from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.classes.samples import CutoffSample
from src.data.classes.settings import EVENTS, TabPfnSettings
from src.models.tabular_model import stack_samples
from src.utils.progress import TRAIN, create_progress

logger = logging.getLogger(__name__)

CONTEXT_FILE = "tabpfn_context_{event}.parquet"
META_FILE = "tabpfn_meta.json"
LABEL_COLUMN = "__label__"


def sample_context(
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    max_rows: int,
    positive_fraction: float,
    seed: int,
) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    positives = np.flatnonzero(labels.to_numpy() == 1)
    negatives = np.flatnonzero(labels.to_numpy() == 0)
    wanted = max(int(max_rows * positive_fraction), int(max_rows * len(positives) / max(len(labels), 1)))
    chosen_positive = generator.choice(positives, size=min(wanted, len(positives)), replace=False)
    chosen_negative = generator.choice(negatives, size=min(max_rows - len(chosen_positive), len(negatives)), replace=False)
    rows = np.sort(np.concatenate([chosen_positive, chosen_negative]))
    context = features.iloc[rows].reset_index(drop=True).copy()
    context[LABEL_COLUMN] = labels.iloc[rows].to_numpy()
    return context


class TabPfnEventModel:
    def __init__(self, *, settings: TabPfnSettings) -> None:
        self._settings = settings
        self._contexts: dict[str, pd.DataFrame] = {}
        self._classifiers: dict[str, object] = {}
        self._columns: list[str] = []

    @property
    def name(self) -> str:
        return f"tabpfn_{self._settings.model_version}"

    def fit(
        self,
        *,
        train: Sequence[CutoffSample],
        calibration: Sequence[CutoffSample],
    ) -> None:
        recent = train[-self._settings.recent_weeks :] if self._settings.recent_weeks else train
        features, labels = stack_samples(samples=recent)
        self._columns = list(features.columns)
        for index, event in enumerate(EVENTS):
            self._contexts[event] = sample_context(
                features=features,
                labels=labels[event],
                max_rows=self._settings.max_context_rows,
                positive_fraction=self._settings.positive_fraction,
                seed=self._settings.seed + index,
            )
            logger.info(
                "🧠 %s context for %s: %d rows from %d train weeks (%.1f%% positive)",
                self.name,
                event,
                len(self._contexts[event]),
                len(recent),
                100 * self._contexts[event][LABEL_COLUMN].mean(),
            )
        self._fit_classifiers()

    def predict_proba(self, *, samples: Sequence[CutoffSample]) -> np.ndarray:
        features, _ = stack_samples(samples=samples)
        missing = sorted(set(self._columns) - set(features.columns))
        if missing:
            raise ValueError(f"Features missing at prediction time: {missing[:10]}")
        values = features[self._columns].to_numpy(dtype="float32")
        probabilities = np.zeros((len(values), len(EVENTS)), dtype="float64")
        batch = self._settings.predict_batch_rows
        for index, event in enumerate(EVENTS):
            classifier = self._classifiers[event]
            parts = [classifier.predict_proba(values[start : start + batch])[:, 1] for start in range(0, len(values), batch)]
            probabilities[:, index] = np.concatenate(parts)
        return probabilities

    def feature_importance(self) -> pd.DataFrame | None:
        return None

    def save(self, *, model_dir: Path) -> None:
        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        for event, context in self._contexts.items():
            context.to_parquet(model_dir / CONTEXT_FILE.format(event=event), index=False)
        (model_dir / META_FILE).write_text(json.dumps({"columns": self._columns}, indent=2), encoding="utf-8")

    def load(self, *, model_dir: Path) -> None:
        self._columns = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))["columns"]
        self._contexts = {event: pd.read_parquet(model_dir / CONTEXT_FILE.format(event=event)) for event in EVENTS}
        self._fit_classifiers()
        logger.info("Loaded %s contexts from %s", self.name, model_dir)

    def _fit_classifiers(self) -> None:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        version = ModelVersion(self._settings.model_version)
        with create_progress(
            stage=TRAIN,
            total=len(EVENTS),
            unit="event",
        ) as bar:
            for event in EVENTS:
                bar.set_postfix_str(event)
                context = self._contexts[event]
                classifier = TabPFNClassifier.create_default_for_version(
                    version,
                    device=self._settings.device,
                    n_estimators=self._settings.n_estimators,
                    random_state=self._settings.seed,
                    ignore_pretraining_limits=True,
                )
                classifier.fit(
                    context[self._columns].to_numpy(dtype="float32"),
                    context[LABEL_COLUMN].to_numpy(dtype="int64"),
                )
                self._classifiers[event] = classifier
                bar.update(1)
