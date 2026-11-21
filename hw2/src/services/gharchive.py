from __future__ import annotations

import logging
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.progress import GH_ARCHIVE, create_progress

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://data.gharchive.org"
HOUR_FILE_SUFFIX = ".json.gz"
PARTIAL_SUFFIX = ".part"
WGET_SERVER_ERROR = 8
NOT_FOUND_STATUS = " 404 "
REQUEST_TIMEOUT_SECONDS = 120
BYTES_PER_GB = 1 << 30
MAX_LOGGED_HOURS = 20
SAVED = "saved"
MISSING = "missing"
FAILED = "failed"


class HourNotFoundError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class DownloadSummary:
    saved: tuple[str, ...]
    skipped: tuple[str, ...]
    missing: tuple[str, ...]
    failed: tuple[str, ...]


def hour_file_name(*, hour: pd.Timestamp) -> str:
    return f"{hour:%Y-%m-%d}-{hour.hour}{HOUR_FILE_SUFFIX}"


def parse_hour_file_name(*, name: str) -> pd.Timestamp:
    stem = name.removesuffix(HOUR_FILE_SUFFIX)
    day, hour = stem.rsplit("-", 1)
    return pd.Timestamp(day) + pd.Timedelta(hours=int(hour))


def list_hour_files(*, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    return [hour_file_name(hour=hour) for hour in pd.date_range(start, end, freq="h", inclusive="left")]


def current_hour() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")


class GHArchiveDownloader:
    def __init__(
        self,
        *,
        raw_dir: Path,
        workers: int,
        retries: int,
        min_free_gb: float,
    ) -> None:
        self._raw_dir = raw_dir
        self._workers = workers
        self._retries = retries
        self._min_free_gb = min_free_gb

    def download(self, *, start: pd.Timestamp, end: pd.Timestamp) -> DownloadSummary:
        self._raw_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        names = list_hour_files(start=start, end=min(end, current_hour()))
        skipped = [name for name in names if (self._raw_dir / name).is_file()]
        pending = [name for name in names if not (self._raw_dir / name).is_file()]
        logger.info("📦 %d hours requested, %d already saved, %d to download with %d workers", len(names), len(skipped), len(pending), self._workers)
        results = {SAVED: [], MISSING: [], FAILED: []}
        with ThreadPoolExecutor(max_workers=self._workers) as pool, create_progress(stage=GH_ARCHIVE, total=len(pending), unit="hour") as bar:
            futures = {pool.submit(self._fetch, name): name for name in pending}
            for future in as_completed(futures):
                results[future.result()].append(futures[future])
                bar.update(1)
        summary = DownloadSummary(
            saved=tuple(sorted(results[SAVED])),
            skipped=tuple(skipped),
            missing=tuple(sorted(results[MISSING])),
            failed=tuple(sorted(results[FAILED])),
        )
        logger.info("✅ %d saved, %d skipped, %d not on GH Archive, %d failed", len(summary.saved), len(summary.skipped), len(summary.missing), len(summary.failed))
        if summary.missing:
            logger.warning("🚧 Not on GH Archive: %s", ", ".join(summary.missing[:MAX_LOGGED_HOURS]))
        if summary.failed:
            raise RuntimeError(f"{len(summary.failed)} hours failed, run the download again: {', '.join(summary.failed[:MAX_LOGGED_HOURS])}")
        return summary

    def _fetch(self, name: str) -> str:
        for attempt in range(1, self._retries + 1):
            try:
                self._check_free_space()
                self._save(name=name)
                return SAVED
            except HourNotFoundError:
                return MISSING
            except (OSError, subprocess.SubprocessError) as error:
                logger.warning("🚧 %s attempt %d: %s", name, attempt, error)
            time.sleep(attempt)
        return FAILED

    def _check_free_space(self) -> None:
        free_gb = shutil.disk_usage(self._raw_dir).free / BYTES_PER_GB
        if free_gb < self._min_free_gb:
            raise OSError(f"{self._raw_dir} has {free_gb:.1f} GB free, need {self._min_free_gb:.1f} GB")

    def _save(self, *, name: str) -> None:
        target = self._raw_dir / name
        partial = target.with_name(name + PARTIAL_SUFFIX)
        result = subprocess.run(
            [
                "wget",
                "--quiet",
                "--server-response",
                "--tries=1",
                f"--timeout={REQUEST_TIMEOUT_SECONDS}",
                f"--output-document={partial}",
                f"{ARCHIVE_URL}/{name}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            partial.unlink(missing_ok=True)
            if result.returncode == WGET_SERVER_ERROR and NOT_FOUND_STATUS in result.stderr:
                raise HourNotFoundError(name)
            raise OSError(f"wget exit code {result.returncode}: {result.stderr.strip()[-200:]}")
        partial.rename(target)
