from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from src.data.classes.settings import DataSettings

logger = logging.getLogger(__name__)

EXPECTED_README_TITLE = "# BigData2026-Assignment2"


def read_local_head(*, folder: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(folder), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def read_remote_head(*, repository_url: str) -> str:
    result = subprocess.run(
        ["git", "ls-remote", repository_url, "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.split()[0] if result.stdout.strip() else ""


def read_commit_time(*, folder: Path, revision: str = "HEAD") -> str:
    result = subprocess.run(
        ["git", "-C", str(folder), "log", "-1", "--format=%cd", "--date=format:%Y%m%d_%H%M", revision],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


class DatasetDownloader:
    def __init__(self, *, settings: DataSettings) -> None:
        self._settings = settings

    def download(self) -> Path:
        target = self._settings.dataset_dir
        if target.exists() and read_local_head(folder=target) == read_remote_head(repository_url=self._settings.repository_url):
            logger.info("✅ %s is already the latest TA repository %s, nothing to download", target, read_commit_time(folder=target))
            return target
        if target.exists():
            self._remove_previous(target=target)
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        logger.info("📥 Cloning %s into %s", self._settings.repository_url, target)
        subprocess.run(
            ["git", "clone", "--depth", "1", self._settings.repository_url, str(target)],
            check=True,
        )
        logger.info("✅ TA repository %s", read_commit_time(folder=target))
        return target

    def _remove_previous(self, *, target: Path) -> None:
        readme = target / "README.md"
        if not (target / ".git").is_dir() or not readme.is_file() or not readme.read_text(encoding="utf-8").startswith(EXPECTED_README_TITLE):
            raise RuntimeError(f"{target} does not look like the TA repository clone. Remove it by hand if it is safe.")
        shutil.rmtree(target)
        logger.info("🗑️ Removed the previous TA repository %s", target)
