from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.tabpfn_model import LABEL_COLUMN, sample_context


def test_sample_context_keeps_size_and_boosts_positives() -> None:
    generator = np.random.default_rng(0)
    features = pd.DataFrame({"a": generator.random(5000), "b": generator.random(5000)})
    labels = pd.Series((generator.random(5000) < 0.03).astype(int))
    context = sample_context(features=features, labels=labels, max_rows=1000, positive_fraction=0.25, seed=0)
    assert len(context) == 1000
    assert context[LABEL_COLUMN].sum() == labels.sum()
    assert context[LABEL_COLUMN].mean() > labels.mean()
    assert list(context.columns) == ["a", "b", LABEL_COLUMN]


def test_sample_context_small_table_uses_all_rows() -> None:
    features = pd.DataFrame({"a": range(50)})
    labels = pd.Series([1] * 5 + [0] * 45)
    context = sample_context(features=features, labels=labels, max_rows=1000, positive_fraction=0.25, seed=0)
    assert len(context) == 50
