from __future__ import annotations

from pathlib import Path

import pytest

from src.data.classes.settings import ExperimentSettings
from src.services.config_loader import ConfigError, build_section, load_experiment_settings, load_graph_settings

STORAGE_VARIABLE = "BIG_DATA_NYCU_STORAGE"
STORAGE = Path("/mnt/storage")
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENT_CONFIGS = sorted(path for path in CONFIG_ROOT.glob("*/*.yaml") if path.parent.name != "graphs")
Q1_GITHUB_USER_IDS = (22996444, 28983449)


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(STORAGE_VARIABLE, str(STORAGE))
    return STORAGE


@pytest.fixture
def config_with_unknown_key() -> dict:
    return {
        "name": "inline",
        "data": {"dataset_dir": "/data"},
        "model": {"depth": 3},
    }


def test_unknown_key_is_rejected(config_with_unknown_key: dict) -> None:
    with pytest.raises(ConfigError, match="Unknown keys"):
        build_section(
            section_type=ExperimentSettings,
            values=config_with_unknown_key,
            source=Path("inline"),
        )


@pytest.mark.parametrize("config", EXPERIMENT_CONFIGS, ids=lambda path: f"{path.parent.name}/{path.name}")
def test_shipped_experiment_config_reads_data_from_storage(storage: Path, config: Path) -> None:
    settings = load_experiment_settings(path=config)
    assert settings.data.dataset_dir.is_relative_to(storage / "data/hw2")


def test_missing_storage_variable_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(STORAGE_VARIABLE, raising=False)
    with pytest.raises(ConfigError, match="Unresolved environment variable"):
        load_experiment_settings(path=CONFIG_ROOT / "xgboost/exp1.yaml")


def test_graph_config_lists_q1_users(storage: Path) -> None:
    settings = load_graph_settings(path=CONFIG_ROOT / "graphs/render.yaml")
    assert settings.github_user_ids == Q1_GITHUB_USER_IDS
