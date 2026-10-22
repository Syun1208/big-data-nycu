from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.classes.settings import EVENTS
from src.services.ensemble import load_split, simplex_grid


def write_split(*, path, users, seed) -> None:
    generator = np.random.default_rng(seed)
    frame = pd.DataFrame({"user_id": users, "github_user_id": users, "cutoff": "2026-09-30 16:00", "split": "calibration"})
    for event in EVENTS:
        frame[f"proba_{event}"] = generator.random(len(users))
        frame[f"label_{event}"] = generator.integers(0, 2, len(users))
    frame.to_parquet(path, index=False)


def test_simplex_grid_sums_to_one() -> None:
    grid = simplex_grid(size=3, step=0.25)
    assert len(grid) == 15
    assert all(sum(weights) == pytest.approx(1.0) for weights in grid)


def test_load_split_aligns_members(tmp_path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    write_split(path=first / "split.parquet", users=[3, 1, 2], seed=0)
    write_split(path=second / "split.parquet", users=[1, 2, 3], seed=1)
    probabilities, truth, keys = load_split(run_dirs=[first, second], name="split.parquet")
    assert probabilities.shape == (2, 3, len(EVENTS))
    assert keys["github_user_id"].tolist() == [1, 2, 3]
    assert truth.shape == (3, len(EVENTS))


def test_load_split_rejects_different_users(tmp_path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    write_split(path=first / "split.parquet", users=[1, 2], seed=0)
    write_split(path=second / "split.parquet", users=[1, 3], seed=1)
    with pytest.raises(ValueError):
        load_split(run_dirs=[first, second], name="split.parquet")
