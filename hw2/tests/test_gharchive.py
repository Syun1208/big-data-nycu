from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from src.data.classes.settings import DataSettings
from src.services import gharchive
from src.services.gharchive import GHArchiveDownloader, hour_file_name, list_hour_files, parse_hour_file_name
from src.services.preprocessing import MANIFEST_FILE, SELECTED_DIR, TablePreprocessor, build_manifest

START = pd.Timestamp("2026-10-31 22:00:00")
END = pd.Timestamp("2026-11-01 01:00:00")
LATER_END = pd.Timestamp("2026-11-01 02:00:00")
HOURS = ["2026-10-31-22.json.gz", "2026-10-31-23.json.gz", "2026-11-01-0.json.gz"]
CONTENT = b"events"
OUTPUT_OPTION = "--output-document="


def fake_wget(*, returncode: int = 0, stderr: str = "") -> Callable[..., subprocess.CompletedProcess]:
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        output = next(part for part in command if part.startswith(OUTPUT_OPTION)).removeprefix(OUTPUT_OPTION)
        Path(output).write_bytes(CONTENT)
        return subprocess.CompletedProcess(command, returncode, stderr=stderr)

    return run


@pytest.fixture
def downloader(tmp_path: Path) -> GHArchiveDownloader:
    return GHArchiveDownloader(raw_dir=tmp_path, workers=2, retries=2, min_free_gb=0.0)


@pytest.fixture
def no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gharchive.time, "sleep", lambda seconds: None)


@pytest.fixture
def requested(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    urls: list[str] = []
    found = fake_wget()
    not_found = fake_wget(returncode=8, stderr="  HTTP/1.1 404 Not Found")

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        urls.append(command[-1])
        return (not_found if command[-1].endswith(HOURS[2]) else found)(command, **kwargs)

    monkeypatch.setattr(gharchive.subprocess, "run", run)
    return urls


@pytest.fixture
def settings(tmp_path: Path) -> DataSettings:
    return DataSettings(dataset_dir=tmp_path / "assignment2")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "raw"
    folder.mkdir()
    for name in HOURS:
        (folder / name).write_bytes(CONTENT)
    return folder


@pytest.fixture
def preprocess_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []
    monkeypatch.setattr("src.services.preprocessing.subprocess.run", lambda command, check: calls.append(command))
    return calls


@pytest.mark.parametrize(
    ("hour", "name"),
    [
        pytest.param("2026-11-01 00:00:00", "2026-11-01-0.json.gz", id="single-digit-hour"),
        pytest.param("2026-11-01 23:00:00", "2026-11-01-23.json.gz", id="two-digit-hour"),
    ],
)
def test_hour_file_name_round_trip(hour: str, name: str) -> None:
    assert hour_file_name(hour=pd.Timestamp(hour)) == name
    assert parse_hour_file_name(name=name) == pd.Timestamp(hour)


def test_hour_range_excludes_the_end() -> None:
    assert list_hour_files(start=START, end=END) == HOURS


def test_download_skips_saved_hours(downloader: GHArchiveDownloader, requested: list[str], tmp_path: Path) -> None:
    (tmp_path / HOURS[0]).write_bytes(CONTENT)
    summary = downloader.download(start=START, end=END)
    assert summary.skipped == (HOURS[0],)
    assert not any(url.endswith(HOURS[0]) for url in requested)


def test_download_reports_hours_not_on_archive(downloader: GHArchiveDownloader, requested: list[str], tmp_path: Path) -> None:
    summary = downloader.download(start=START, end=END)
    assert summary.missing == (HOURS[2],)
    assert not (tmp_path / HOURS[2]).exists()


def test_download_does_not_request_future_hours(
    downloader: GHArchiveDownloader,
    requested: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gharchive, "current_hour", lambda: START + pd.Timedelta(hours=1))
    downloader.download(start=START, end=END)
    assert len(requested) == 1


def test_interrupted_download_is_not_kept(
    downloader: GHArchiveDownloader,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    no_wait: None,
) -> None:
    monkeypatch.setattr(gharchive.subprocess, "run", fake_wget(returncode=4, stderr="connection reset"))
    with pytest.raises(RuntimeError, match="3 hours failed"):
        downloader.download(start=START, end=END)
    assert not list(tmp_path.glob("*.json.gz*"))


def test_forbidden_is_a_failure_not_a_missing_hour(
    downloader: GHArchiveDownloader,
    monkeypatch: pytest.MonkeyPatch,
    no_wait: None,
) -> None:
    monkeypatch.setattr(gharchive.subprocess, "run", fake_wget(returncode=8, stderr="  HTTP/1.1 403 Forbidden"))
    with pytest.raises(RuntimeError, match="3 hours failed"):
        downloader.download(start=START, end=END)


def test_download_stops_when_the_disk_is_almost_full(requested: list[str], tmp_path: Path, no_wait: None) -> None:
    downloader = GHArchiveDownloader(raw_dir=tmp_path, workers=1, retries=1, min_free_gb=float("inf"))
    with pytest.raises(RuntimeError, match="failed"):
        downloader.download(start=START, end=END)
    assert requested == []


def test_manifest_lists_hours_missing_inside_the_period() -> None:
    manifest = build_manifest(names=[HOURS[0], HOURS[2]])
    assert manifest["missing_hours"] == [HOURS[1]]
    assert manifest["first_hour_utc"] == str(START)


def test_preprocess_runs_the_ta_script_and_writes_the_manifest(
    settings: DataSettings,
    raw_dir: Path,
    preprocess_calls: list[list[str]],
) -> None:
    settings.tables_dir.mkdir(parents=True)
    TablePreprocessor(settings=settings, raw_dir=raw_dir, start=START, end=LATER_END).run()
    assert preprocess_calls[0][1] == str(settings.preprocess_script)
    assert preprocess_calls[0][3] == str(settings.tables_dir / SELECTED_DIR)
    assert json.loads((settings.tables_dir / MANIFEST_FILE).read_text(encoding="utf-8"))["hours"] == len(HOURS)


def test_preprocess_skips_when_the_hours_did_not_change(
    settings: DataSettings,
    raw_dir: Path,
    preprocess_calls: list[list[str]],
) -> None:
    settings.tables_dir.mkdir(parents=True)
    processor = TablePreprocessor(settings=settings, raw_dir=raw_dir, start=START, end=LATER_END)
    processor.run()
    processor.run()
    assert len(preprocess_calls) == 1


def test_preprocess_runs_again_when_a_new_hour_arrives(
    settings: DataSettings,
    raw_dir: Path,
    preprocess_calls: list[list[str]],
) -> None:
    settings.tables_dir.mkdir(parents=True)
    processor = TablePreprocessor(settings=settings, raw_dir=raw_dir, start=START, end=LATER_END)
    processor.run()
    (raw_dir / "2026-11-01-1.json.gz").write_bytes(CONTENT)
    processor.run()
    assert len(preprocess_calls) == 2


def test_preprocess_needs_raw_files(settings: DataSettings, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No GH Archive"):
        TablePreprocessor(settings=settings, raw_dir=tmp_path, start=START, end=LATER_END).run()


def test_preprocess_uses_only_hours_before_the_end(
    settings: DataSettings,
    raw_dir: Path,
    preprocess_calls: list[list[str]],
) -> None:
    settings.tables_dir.mkdir(parents=True)
    processor = TablePreprocessor(settings=settings, raw_dir=raw_dir, start=START, end=END - pd.Timedelta(hours=1))
    processor.run()
    assert sorted(path.name for path in processor.selected_dir.iterdir()) == HOURS[:2]
    assert json.loads((settings.tables_dir / MANIFEST_FILE).read_text(encoding="utf-8"))["last_hour_utc"] == "2026-10-31 23:00:00"
