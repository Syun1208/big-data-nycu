from __future__ import annotations

from pathlib import Path

ENV_FILE_NAME = ".env"


def find_env_file(*, start: Path) -> Path | None:
    for directory in (start, *start.parents):
        candidate = directory / ENV_FILE_NAME
        if candidate.is_file():
            return candidate
    return None


def load_env(*, start: Path) -> Path | None:
    from dotenv import load_dotenv

    env_file = find_env_file(start=start)
    if env_file is not None:
        load_dotenv(
            dotenv_path=env_file,
            override=False,
        )
    return env_file
