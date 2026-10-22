from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class EventClassifier(Protocol):
    def fit(
        self,
        *,
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        valid_features: pd.DataFrame,
        valid_labels: pd.Series,
    ) -> None: ...

    def predict_proba(self, *, features: pd.DataFrame) -> np.ndarray: ...

    def feature_importance(self, *, columns: list[str]) -> pd.Series: ...

    def save(self, *, path: Path) -> None: ...

    def load(self, *, path: Path) -> None: ...
