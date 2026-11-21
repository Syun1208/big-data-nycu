from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.data.classes.settings import DataSettings
from src.services.gharchive import HOUR_FILE_SUFFIX, list_hour_files, parse_hour_file_name

logger = logging.getLogger(__name__)

MANIFEST_FILE = "source_hours.json"
SELECTED_DIR = "selected_hours"


def build_manifest(*, names: list[str]) -> dict:
    hours = sorted(parse_hour_file_name(name=name) for name in names)
    first, last = hours[0], hours[-1]
    expected = list_hour_files(start=first, end=last + pd.Timedelta(hours=1))
    return {
        "first_hour_utc": str(first),
        "last_hour_utc": str(last),
        "hours": len(hours),
        "missing_hours": sorted(set(expected) - set(names)),
    }


class TablePreprocessor:
    def __init__(
        self,
        *,
        settings: DataSettings,
        raw_dir: Path,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> None:
        self._settings = settings
        self._raw_dir = raw_dir
        self._start = start
        self._end = end

    @property
    def selected_dir(self) -> Path:
        return self._settings.tables_dir / SELECTED_DIR

    def run(self) -> dict:
        names = [name for name in list_hour_files(start=self._start, end=self._end) if (self._raw_dir / name).is_file()]
        if not names:
            raise FileNotFoundError(f"No GH Archive hour files from {self._start} to {self._end} in {self._raw_dir}")
        manifest = build_manifest(names=names)
        path = self._settings.tables_dir / MANIFEST_FILE
        if path.is_file() and json.loads(path.read_text(encoding="utf-8")) == manifest:
            logger.info("✅ %s already built from these %d hours, nothing to do", self._settings.db_dir, manifest["hours"])
            return manifest
        path.unlink(missing_ok=True)
        self._link_selected(names=names)
        logger.info("🧱 Building tables from %d hours, %s to %s", manifest["hours"], manifest["first_hour_utc"], manifest["last_hour_utc"])
        subprocess.run(
            [sys.executable, str(self._settings.preprocess_script), "--raw", str(self.selected_dir), "--out", str(self._settings.tables_dir)],
            check=True,
        )
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("💾 %s, %d hours missing inside the period", path, len(manifest["missing_hours"]))
        return manifest

    def _link_selected(self, *, names: list[str]) -> None:
        shutil.rmtree(self.selected_dir, ignore_errors=True)
        self.selected_dir.mkdir(parents=True)
        for name in names:
            (self.selected_dir / name).symlink_to((self._raw_dir / name).resolve())
