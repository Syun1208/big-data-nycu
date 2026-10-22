from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tarfile
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from src.services.data_download import read_commit_time
from src.utils.progress import RELEASES, create_progress

logger = logging.getLogger(__name__)

API_ROOT = "https://api.github.com/repos"
EVENT_PAGES = 3
EVENTS_PER_PAGE = 100
RELEASE_PREFIX = "data up to "
DB_PATH = "data/db"
ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class ReleaseCommit:
    sha: str
    pushed_at: str


@dataclass(frozen=True, slots=True)
class ReleaseInfo:
    commit: ReleaseCommit
    message: str
    snapshot_name: str
    repo: Path


def repository_slug(*, repository_url: str) -> str:
    return repository_url.removeprefix("https://github.com/").removesuffix(".git")


def read_json(*, url: str) -> object:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def list_release_commits(*, slug: str) -> list[ReleaseCommit]:
    commits = {}
    for page in range(1, EVENT_PAGES + 1):
        events = read_json(url=f"{API_ROOT}/{slug}/events?per_page={EVENTS_PER_PAGE}&page={page}")
        for event in events:
            if event["type"] == "PushEvent":
                commits[event["payload"]["head"]] = event["created_at"]
        if len(events) < EVENTS_PER_PAGE:
            break
    head = read_json(url=f"{API_ROOT}/{slug}/commits?per_page=1")[0]
    commits.setdefault(head["sha"], head["commit"]["committer"]["date"])
    return sorted((ReleaseCommit(sha=sha, pushed_at=time) for sha, time in commits.items()), key=lambda commit: commit.pushed_at)


def latest_per_release(*, infos: Iterable[ReleaseInfo]) -> list[ReleaseInfo]:
    latest = {}
    for info in sorted(infos, key=lambda item: item.commit.pushed_at):
        if info.message.startswith(RELEASE_PREFIX):
            latest[info.message] = info
    return sorted(latest.values(), key=lambda item: item.message)


def run_git(*, folder: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(folder), *args], capture_output=True, check=True)


def run_parallel(
    *,
    function: Callable[[ItemT], ResultT],
    items: list[ItemT],
    workers: int,
    unit: str,
) -> list[ResultT]:
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool, create_progress(stage=RELEASES, total=len(items), unit=unit) as bar:
        futures = [pool.submit(function, item) for item in items]
        for future in as_completed(futures):
            results.append(future.result())
            bar.update(1)
    return results


class ReleaseHistoryDownloader:
    def __init__(
        self,
        *,
        repository_url: str,
        snapshot_dir: Path,
        work_dir: Path,
        workers: int,
    ) -> None:
        self._repository_url = repository_url
        self._snapshot_dir = snapshot_dir
        self._work_dir = work_dir
        self._workers = workers

    def download(self) -> list[Path]:
        commits = list_release_commits(slug=repository_slug(repository_url=self._repository_url))
        logger.info("📜 %d pushes found in the GitHub events, %d parallel workers", len(commits), self._workers)
        shutil.rmtree(self._work_dir, ignore_errors=True)
        try:
            infos = run_parallel(function=self._read_info, items=commits, workers=self._workers, unit="commit")
            releases = latest_per_release(infos=[info for info in infos if info is not None])
            missing = [info for info in releases if not (self._snapshot_dir / info.snapshot_name).exists()]
            logger.info("📦 %d releases, %d already saved, %d to download", len(releases), len(releases) - len(missing), len(missing))
            written = run_parallel(function=self._extract, items=missing, workers=self._workers, unit="release")
        finally:
            shutil.rmtree(self._work_dir, ignore_errors=True)
        kept = sorted(path.name for path in self._snapshot_dir.iterdir() if path.is_dir() and not path.name.startswith("partial-"))
        logger.info("📦 %d snapshots in %s: %s", len(kept), self._snapshot_dir, ", ".join(kept))
        return written

    def _read_info(self, commit: ReleaseCommit) -> ReleaseInfo | None:
        repo = self._work_dir / commit.sha
        repo.mkdir(
            parents=True,
            exist_ok=True,
        )
        run_git(folder=repo, args=["init", "--quiet", "--bare"])
        run_git(folder=repo, args=["remote", "add", "origin", self._repository_url])
        run_git(folder=repo, args=["config", "remote.origin.promisor", "true"])
        run_git(folder=repo, args=["config", "extensions.partialClone", "origin"])
        try:
            run_git(folder=repo, args=["fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", commit.sha])
        except subprocess.CalledProcessError:
            logger.warning("🚧 %s (pushed %s) is no longer on GitHub, skipped", commit.sha[:10], commit.pushed_at)
            return None
        return ReleaseInfo(
            commit=commit,
            message=run_git(folder=repo, args=["log", "-1", "--format=%s", commit.sha]).stdout.decode().strip(),
            snapshot_name=read_commit_time(folder=repo, revision=commit.sha),
            repo=repo,
        )

    def _extract(self, info: ReleaseInfo) -> Path:
        destination = self._snapshot_dir / info.snapshot_name
        partial = destination.with_name(f"partial-{destination.name}")
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(
            parents=True,
            exist_ok=True,
        )
        error_log = info.repo / "archive.log"
        with error_log.open("wb") as errors:
            archive = subprocess.Popen(
                ["git", "-C", str(info.repo), "archive", "--format=tar", info.commit.sha, DB_PATH],
                stdout=subprocess.PIPE,
                stderr=errors,
            )
            with tarfile.open(fileobj=archive.stdout, mode="r|") as tar:
                tar.extractall(path=partial, filter="data")
            code = archive.wait()
        if code != 0:
            raise RuntimeError(f"git archive failed for {info.commit.sha}: {error_log.read_text(errors='replace')}")
        partial.rename(destination)
        shutil.rmtree(info.repo, ignore_errors=True)
        logger.info("💾 %s (%s) saved to %s", info.message, info.commit.sha[:10], destination)
        return destination
