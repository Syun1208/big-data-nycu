from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from src.data.classes.settings import DataSettings

logger = logging.getLogger(__name__)

EXPECTED_README_TITLE = "# BigData2026-Assignment2"


def read_commit_time(*, folder: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(folder), "log", "-1", "--format=%cd", "--date=format:%Y%m%d_%H%M"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


class DatasetDownloader:
    def __init__(self, *, settings: DataSettings) -> None:
        self._settings = settings

    def download(self, *, snapshot_dir: Path | None) -> Path:
        target = self._settings.dataset_dir
        if target.exists():
            self._remove_previous(
                target=target,
                snapshot_dir=snapshot_dir,
            )
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        logger.info("📥 Cloning %s into %s", self._settings.repository_url, target)
        subprocess.run(
            ["git", "clone", "--depth", "1", self._settings.repository_url, str(target)],
            check=True,
        )
        logger.info("✅ Dataset release %s", read_commit_time(folder=target))
        return target

    def _remove_previous(
        self,
        *,
        target: Path,
        snapshot_dir: Path | None,
    ) -> None:
        readme = target / "README.md"
        if not (target / ".git").is_dir() or not readme.is_file() or not readme.read_text(encoding="utf-8").startswith(EXPECTED_README_TITLE):
            raise RuntimeError(f"{target} does not look like the TA dataset clone. Remove it by hand if it is safe.")
        if snapshot_dir is not None:
            destination = snapshot_dir / read_commit_time(folder=target)
            if not destination.exists():
                shutil.copytree(
                    self._settings.db_dir,
                    destination,
                )
                logger.info("📦 Kept the previous release in %s", destination)
        shutil.rmtree(target)
        logger.info("🗑️ Removed the previous release %s", target)
