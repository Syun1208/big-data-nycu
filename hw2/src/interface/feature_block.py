from __future__ import annotations

from typing import Protocol

import pandas as pd

from src.data.classes.samples import FeatureContext


class FeatureBlock(Protocol):
    @property
    def name(self) -> str: ...

    def build(self, *, context: FeatureContext) -> pd.DataFrame: ...
