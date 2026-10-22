from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from src.data.classes.samples import CutoffSample


class ActivityModel(Protocol):
    @property
    def name(self) -> str: ...

    def fit(
        self,
        *,
        train: Sequence[CutoffSample],
        calibration: Sequence[CutoffSample],
    ) -> None: ...

    def predict_proba(self, *, samples: Sequence[CutoffSample]) -> np.ndarray: ...

    def feature_importance(self) -> pd.DataFrame | None: ...

    def save(self, *, model_dir: Path) -> None: ...

    def load(self, *, model_dir: Path) -> None: ...
