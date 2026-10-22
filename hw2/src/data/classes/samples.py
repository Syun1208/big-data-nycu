from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.data.loaders.relational_db import RelationalDatabase

TRAIN_SPLIT = "train"
CALIBRATION_SPLIT = "calibration"
HOLDOUT_SPLIT = "holdout"
PREDICT_SPLIT = "predict"


@dataclass(frozen=True, slots=True)
class Cutoff:
    time: pd.Timestamp
    feature_time: pd.Timestamp
    population_time: pd.Timestamp
    split: str

    @property
    def label(self) -> str:
        return f"{self.time:%Y-%m-%d %H:%M}"


@dataclass(frozen=True, slots=True)
class FeatureContext:
    user_ids: pd.Index
    feature_time: pd.Timestamp
    known_events: pd.DataFrame
    database: RelationalDatabase


@dataclass(frozen=True, slots=True)
class CutoffDataset:
    features: pd.DataFrame
    labels: pd.DataFrame | None
    meta: pd.DataFrame

    @property
    def feature_columns(self) -> list[str]:
        return list(self.features.columns)


@dataclass(frozen=True, slots=True)
class CutoffSample:
    cutoff: Cutoff
    dataset: CutoffDataset
