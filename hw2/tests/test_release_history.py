from __future__ import annotations

from pathlib import Path

import pytest

from src.services.release_history import ReleaseCommit, ReleaseInfo, latest_per_release, repository_slug

REPOSITORY_URL = "https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git"


def release(*, sha: str, pushed_at: str, message: str) -> ReleaseInfo:
    return ReleaseInfo(
        commit=ReleaseCommit(sha=sha, pushed_at=pushed_at),
        message=message,
        snapshot_name=sha,
        repo=Path(sha),
    )


@pytest.fixture
def pushes() -> list[ReleaseInfo]:
    return [
        release(sha="setup", pushed_at="2026-10-13T12:30:00Z", message="Initial commit"),
        release(sha="oct28-late", pushed_at="2026-10-28T13:01:24Z", message="data up to 2026-10-28T12:00:00Z"),
        release(sha="oct27", pushed_at="2026-10-27T12:42:12Z", message="data up to 2026-10-27T12:00:00Z"),
        release(sha="oct28-early", pushed_at="2026-10-28T12:42:11Z", message="data up to 2026-10-28T12:00:00Z"),
    ]


@pytest.fixture
def releases(pushes: list[ReleaseInfo]) -> list[ReleaseInfo]:
    return latest_per_release(infos=pushes)


def test_one_snapshot_per_data_release(releases: list[ReleaseInfo]) -> None:
    assert [info.commit.sha for info in releases] == ["oct27", "oct28-late"]


def test_commits_without_release_message_are_skipped(releases: list[ReleaseInfo]) -> None:
    assert "setup" not in {info.commit.sha for info in releases}


def test_repository_slug_drops_host_and_suffix() -> None:
    assert repository_slug(repository_url=REPOSITORY_URL) == "nycubigdata-2026fall/BigData2026-Assignment2"
