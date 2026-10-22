from __future__ import annotations

from pathlib import Path

import pytest

from main import latest_run
from src.services.training import THRESHOLDS_FILE

FINISHED_RUN = "2026-10-30_hw2.sthn.exp2.hybrid.short_windows"
UNFINISHED_RUN = "2026-10-31_hw2.sthn.exp2.hybrid.short_windows"


@pytest.fixture
def config(project_root: Path) -> Path:
    return project_root / "configs/sthn/exp2.hybrid.short_windows.yaml"


@pytest.fixture
def model_root(tmp_path: Path) -> Path:
    for name in (FINISHED_RUN, UNFINISHED_RUN):
        (tmp_path / name).mkdir()
    (tmp_path / FINISHED_RUN / THRESHOLDS_FILE).write_text("{}", encoding="utf-8")
    return tmp_path


def test_latest_run_skips_a_run_that_did_not_finish(model_root: Path, config: Path) -> None:
    assert latest_run(model_root=model_root, config=config) == FINISHED_RUN


def test_latest_run_fails_when_no_run_finished(tmp_path: Path, config: Path) -> None:
    (tmp_path / UNFINISHED_RUN).mkdir()
    with pytest.raises(FileNotFoundError, match="No finished run"):
        latest_run(model_root=tmp_path, config=config)
