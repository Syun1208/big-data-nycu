from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.classes.settings import EVENTS
from src.services.benchmark import current_data_runs, holdout_week
from src.services.ensemble import load_split, read_member_metrics, simplex_grid
from src.services.training import METRICS_FILE

SPLIT_FILE = "split.parquet"
QUARTER = 0.25
THREE_MEMBER_QUARTER_GRID_SIZE = 15
NEW_DATA_END = "2026-11-21 11:59:59"
OLD_DATA_END = "2026-10-30 12:00:00"


def write_split(*, path: Path, users: Sequence[int], seed: int) -> None:
    generator = np.random.default_rng(seed)
    frame = pd.DataFrame({"user_id": users, "github_user_id": users, "cutoff": "2026-09-30 16:00", "split": "calibration"})
    for event in EVENTS:
        frame[f"proba_{event}"] = generator.random(len(users))
        frame[f"label_{event}"] = generator.integers(0, 2, len(users))
    frame.to_parquet(path, index=False)


@pytest.fixture
def grid() -> list[tuple[float, ...]]:
    return simplex_grid(size=3, step=QUARTER)


@pytest.fixture
def member_dirs(tmp_path: Path) -> list[Path]:
    folders = [tmp_path / "first", tmp_path / "second"]
    for folder in folders:
        folder.mkdir()
    return folders


@pytest.fixture
def aligned_split(member_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    write_split(path=member_dirs[0] / SPLIT_FILE, users=[3, 1, 2], seed=0)
    write_split(path=member_dirs[1] / SPLIT_FILE, users=[1, 2, 3], seed=1)
    return load_split(run_dirs=member_dirs, name=SPLIT_FILE)


def test_simplex_grid_has_every_split_into_quarters(grid: list[tuple[float, ...]]) -> None:
    assert len(grid) == THREE_MEMBER_QUARTER_GRID_SIZE


def test_simplex_grid_weights_sum_to_one(grid: list[tuple[float, ...]]) -> None:
    np.testing.assert_allclose(np.sum(grid, axis=1), 1.0)


def test_members_are_stacked(aligned_split: tuple[np.ndarray, np.ndarray, pd.DataFrame]) -> None:
    probabilities, _, _ = aligned_split
    assert probabilities.shape == (2, 3, len(EVENTS))


def test_rows_are_aligned_by_user(aligned_split: tuple[np.ndarray, np.ndarray, pd.DataFrame]) -> None:
    _, _, keys = aligned_split
    assert keys["github_user_id"].tolist() == [1, 2, 3]


def test_truth_has_one_column_per_event(aligned_split: tuple[np.ndarray, np.ndarray, pd.DataFrame]) -> None:
    _, truth, _ = aligned_split
    assert truth.shape == (3, len(EVENTS))


def test_members_with_different_users_are_rejected(member_dirs: list[Path]) -> None:
    write_split(path=member_dirs[0] / SPLIT_FILE, users=[1, 2], seed=0)
    write_split(path=member_dirs[1] / SPLIT_FILE, users=[1, 3], seed=1)
    with pytest.raises(ValueError, match="different users"):
        load_split(run_dirs=member_dirs, name=SPLIT_FILE)


def write_metrics(*, folder: Path, data_end: str) -> None:
    payload = {"data_end": data_end, "cutoffs": {"2026-10-31 16:00": "calibration", "2026-11-07 16:00": "holdout"}}
    (folder / METRICS_FILE).write_text(json.dumps(payload), encoding="utf-8")


def test_members_trained_on_different_data_are_rejected(member_dirs: list[Path]) -> None:
    write_metrics(folder=member_dirs[0], data_end=NEW_DATA_END)
    write_metrics(folder=member_dirs[1], data_end=OLD_DATA_END)
    with pytest.raises(ValueError, match="different data"):
        read_member_metrics(run_dirs=member_dirs)


def test_benchmark_keeps_runs_on_the_latest_data(member_dirs: list[Path]) -> None:
    write_metrics(folder=member_dirs[0], data_end=NEW_DATA_END)
    write_metrics(folder=member_dirs[1], data_end=OLD_DATA_END)
    assert current_data_runs(run_dirs=member_dirs) == [member_dirs[0]]


def test_holdout_week_is_read_from_the_run(member_dirs: list[Path]) -> None:
    write_metrics(folder=member_dirs[0], data_end=NEW_DATA_END)
    assert holdout_week(run_dir=member_dirs[0]) == "2026-11-07 16:00"
