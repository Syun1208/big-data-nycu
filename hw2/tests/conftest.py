from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from src.data.loaders.event_log import EventLog, build_event_log
from src.data.loaders.relational_db import RelationalDatabase
from src.utils.env import load_env
from tests.factories import build_database

ROOT = Path(__file__).resolve().parents[1]
STORAGE_VARIABLE = "BIG_DATA_NYCU_STORAGE"
EVALUATION_SUBDIR = Path("data/hw2/assignment2/evaluation")


def pytest_configure(config: pytest.Config) -> None:
    load_env(start=ROOT)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture
def database() -> RelationalDatabase:
    return build_database()


@pytest.fixture
def database_factory() -> Callable[..., RelationalDatabase]:
    return build_database


@pytest.fixture
def event_log(database: RelationalDatabase) -> EventLog:
    return build_event_log(database=database)


@pytest.fixture(scope="session")
def evaluation_dir() -> Path:
    folder = Path(os.environ.get(STORAGE_VARIABLE, "/nonexistent")) / EVALUATION_SUBDIR
    if not (folder / "evaluate.py").is_file():
        pytest.skip("TA evaluate.py not downloaded, run bash scripts/download_data.sh")
    return folder


@pytest.fixture(scope="session")
def ta_evaluate(evaluation_dir: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("ta_evaluate", evaluation_dir / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
