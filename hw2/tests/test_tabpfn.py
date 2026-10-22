from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.tabpfn_model import LABEL_COLUMN, sample_context

LARGE_ROWS = 5000
SMALL_ROWS = 50
SMALL_POSITIVES = 5
POSITIVE_RATE = 0.03
MAX_ROWS = 1000
POSITIVE_FRACTION = 0.25


@pytest.fixture
def imbalanced_features() -> pd.DataFrame:
    generator = np.random.default_rng(0)
    return pd.DataFrame({"a": generator.random(LARGE_ROWS), "b": generator.random(LARGE_ROWS)})


@pytest.fixture
def imbalanced_labels() -> pd.Series:
    generator = np.random.default_rng(1)
    return pd.Series((generator.random(LARGE_ROWS) < POSITIVE_RATE).astype(int))


@pytest.fixture
def large_context(imbalanced_features: pd.DataFrame, imbalanced_labels: pd.Series) -> pd.DataFrame:
    return sample_context(
        features=imbalanced_features,
        labels=imbalanced_labels,
        max_rows=MAX_ROWS,
        positive_fraction=POSITIVE_FRACTION,
        seed=0,
    )


@pytest.fixture
def small_context() -> pd.DataFrame:
    return sample_context(
        features=pd.DataFrame({"a": range(SMALL_ROWS)}),
        labels=pd.Series([1] * SMALL_POSITIVES + [0] * (SMALL_ROWS - SMALL_POSITIVES)),
        max_rows=MAX_ROWS,
        positive_fraction=POSITIVE_FRACTION,
        seed=0,
    )


def test_context_has_max_rows(large_context: pd.DataFrame) -> None:
    assert len(large_context) == MAX_ROWS


def test_context_keeps_every_positive(large_context: pd.DataFrame, imbalanced_labels: pd.Series) -> None:
    assert large_context[LABEL_COLUMN].sum() == imbalanced_labels.sum()


def test_context_raises_the_positive_rate(large_context: pd.DataFrame, imbalanced_labels: pd.Series) -> None:
    assert large_context[LABEL_COLUMN].mean() > imbalanced_labels.mean()


def test_context_keeps_feature_columns_then_label(large_context: pd.DataFrame) -> None:
    assert list(large_context.columns) == ["a", "b", LABEL_COLUMN]


def test_small_table_uses_all_rows(small_context: pd.DataFrame) -> None:
    assert len(small_context) == SMALL_ROWS
