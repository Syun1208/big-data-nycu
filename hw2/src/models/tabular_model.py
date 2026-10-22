from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.classes.samples import CutoffSample
from src.data.classes.settings import EVENTS
from src.interface.event_classifier import EventClassifier

logger = logging.getLogger(__name__)

FEATURE_COLUMNS_FILE = "feature_columns.json"
MODEL_SUFFIX = ".ubj"


def stack_samples(*, samples: Sequence[CutoffSample]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    features = pd.concat([sample.dataset.features for sample in samples], ignore_index=True)
    labels = [sample.dataset.labels for sample in samples]
    if any(label is None for label in labels):
        return features, None
    return features, pd.concat(labels, ignore_index=True)


class TabularEventModel:
    def __init__(self, *, classifier_factory: Callable[[str], EventClassifier]) -> None:
        self._classifier_factory = classifier_factory
        self._classifiers: dict[str, EventClassifier] = {}
        self._columns: list[str] = []

    @property
    def name(self) -> str:
        return "xgboost"

    def fit(
        self,
        *,
        train: Sequence[CutoffSample],
        calibration: Sequence[CutoffSample],
    ) -> None:
        train_features, train_labels = stack_samples(samples=train)
        valid_features, valid_labels = stack_samples(samples=calibration)
        self._columns = list(train_features.columns)
        for event in EVENTS:
            classifier = self._classifier_factory(event)
            classifier.fit(
                train_features=train_features,
                train_labels=train_labels[event],
                valid_features=valid_features[self._columns],
                valid_labels=valid_labels[event],
            )
            self._classifiers[event] = classifier

    def predict_proba(self, *, samples: Sequence[CutoffSample]) -> np.ndarray:
        features, _ = stack_samples(samples=samples)
        missing = sorted(set(self._columns) - set(features.columns))
        if missing:
            raise ValueError(f"Features missing at prediction time: {missing[:10]}")
        features = features[self._columns]
        probabilities = np.zeros((len(features), len(EVENTS)), dtype="float64")
        for index, event in enumerate(EVENTS):
            probabilities[:, index] = self._classifiers[event].predict_proba(features=features)
        return probabilities

    def feature_importance(self) -> pd.DataFrame | None:
        return pd.DataFrame(
            {event: classifier.feature_importance(columns=self._columns) for event, classifier in self._classifiers.items()}
        )

    def save(self, *, model_dir: Path) -> None:
        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        (model_dir / FEATURE_COLUMNS_FILE).write_text(json.dumps(self._columns, indent=2), encoding="utf-8")
        for event, classifier in self._classifiers.items():
            classifier.save(path=model_dir / f"{event}{MODEL_SUFFIX}")

    def load(self, *, model_dir: Path) -> None:
        self._columns = json.loads((model_dir / FEATURE_COLUMNS_FILE).read_text(encoding="utf-8"))
        for event in EVENTS:
            classifier = self._classifier_factory(event)
            classifier.load(path=model_dir / f"{event}{MODEL_SUFFIX}")
            self._classifiers[event] = classifier
        logger.info("Loaded %d XGBoost classifiers from %s", len(EVENTS), model_dir)
