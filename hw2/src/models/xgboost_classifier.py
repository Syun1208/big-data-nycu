from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.classes.settings import ModelSettings

logger = logging.getLogger(__name__)


class XGBoostEventClassifier:
    def __init__(
        self,
        *,
        settings: ModelSettings,
        event: str,
    ) -> None:
        self._settings = settings
        self._event = event
        self._model = None

    @property
    def best_iteration(self) -> int | None:
        if self._model is None:
            return None
        return getattr(self._model, "best_iteration", None)

    def fit(
        self,
        *,
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        valid_features: pd.DataFrame,
        valid_labels: pd.Series,
    ) -> None:
        from xgboost import XGBClassifier

        positives = float(train_labels.sum())
        negatives = float(len(train_labels) - positives)
        settings = self._settings
        self._model = XGBClassifier(
            n_estimators=settings.n_estimators,
            learning_rate=settings.learning_rate,
            max_depth=settings.max_depth,
            min_child_weight=settings.min_child_weight,
            subsample=settings.subsample,
            colsample_bytree=settings.colsample_bytree,
            reg_lambda=settings.reg_lambda,
            early_stopping_rounds=settings.early_stopping_rounds,
            tree_method="hist",
            device=settings.device,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=settings.seed,
            n_jobs=4,
        )
        self._model.fit(
            train_features,
            train_labels,
            eval_set=[(valid_features, valid_labels)],
            verbose=False,
        )
        logger.info(
            "🌲 %s: %d train rows (%.1f%% positive, neg/pos %.1f), best iteration %s",
            self._event,
            len(train_labels),
            100 * positives / max(len(train_labels), 1),
            negatives / max(positives, 1.0),
            self.best_iteration,
        )

    def predict_proba(self, *, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"Model for {self._event} is not trained or loaded")
        self._model.set_params(device="cpu")
        return self._model.predict_proba(features)[:, 1]

    def feature_importance(self, *, columns: list[str]) -> pd.Series:
        if self._model is None:
            raise RuntimeError(f"Model for {self._event} is not trained or loaded")
        booster = self._model.get_booster()
        gains = booster.get_score(importance_type="gain")
        return pd.Series(gains, dtype="float64").reindex(columns).fillna(0.0)

    def save(self, *, path: Path) -> None:
        if self._model is None:
            raise RuntimeError(f"Model for {self._event} is not trained")
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._model.save_model(path)

    def load(self, *, path: Path) -> None:
        from xgboost import XGBClassifier

        model = XGBClassifier(device=self._settings.device)
        model.load_model(path)
        self._model = model
