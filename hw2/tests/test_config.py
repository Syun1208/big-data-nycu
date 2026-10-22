from __future__ import annotations

from pathlib import Path

import pytest

from src.data.classes.settings import ExperimentSettings
from src.services.config_loader import ConfigError, build_section, load_experiment_settings, load_graph_settings

ROOT = Path(__file__).resolve().parents[1]


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError):
        build_section(
            section_type=ExperimentSettings,
            values={"name": "x", "data": {"dataset_dir": "/data"}, "model": {"depth": 3}},
            source=Path("inline"),
        )


def test_shipped_configs_load(monkeypatch) -> None:
    monkeypatch.setenv("BIG_DATA_NYCU_STORAGE", "/mnt/storage")
    experiment = load_experiment_settings(path=ROOT / "configs/xgboost/exp1.yaml")
    assert experiment.data.dataset_dir == Path("/mnt/storage/data/hw2/assignment2")
    assert experiment.features.windows_days == (7, 14, 28, 56, 91)
    graphs = load_graph_settings(path=ROOT / "configs/graphs/render.yaml")
    assert graphs.github_user_ids == (22996444, 28983449)
