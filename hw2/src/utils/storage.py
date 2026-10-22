from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

STORAGE_ROOT_VARIABLE = "STORAGE_ROOT"
MOUNT_POINTS = (Path("/mnt"), Path("/media"))
STORAGE_USERS = ("longpm", "Long")
STORAGE_FOLDERS = ("data", "weights", "models", "outputs", "logs", "wandb", "runs", "cache")


class StorageNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProjectStorage:
    root: Path

    def folder(self, *, name: str) -> Path:
        return self.root / name


def storage_candidates() -> list[Path]:
    candidates = []
    for mount in MOUNT_POINTS:
        for user in STORAGE_USERS:
            candidates.extend(mount.glob(f"*/{user}"))
            candidates.extend(path for path in mount.glob(f"{user}/*") if path.is_mount())
    return [path for path in candidates if path.is_dir() and os.access(path, os.W_OK)]


def find_storage_root() -> Path:
    configured = os.environ.get(STORAGE_ROOT_VARIABLE)
    if configured:
        return Path(configured)
    candidates = storage_candidates()
    if not candidates:
        raise StorageNotFoundError(
            f"No writable longpm/Long folder under {', '.join(map(str, MOUNT_POINTS))}. "
            f"Set {STORAGE_ROOT_VARIABLE} in .env."
        )
    return max(candidates, key=lambda path: shutil.disk_usage(path).free)


def prepare_storage(*, repo_root: Path) -> ProjectStorage:
    root = find_storage_root() / repo_root.resolve().name
    resolved = root.resolve()
    if resolved.is_relative_to(Path.home()) or resolved.is_relative_to(repo_root.resolve()):
        raise StorageNotFoundError(f"Storage {root} must live on a /mnt or /media disk, not in /home or the repo.")
    for name in STORAGE_FOLDERS:
        (root / name).mkdir(
            parents=True,
            exist_ok=True,
        )
    os.environ.setdefault("WANDB_DIR", str(root / "wandb"))
    os.environ.setdefault("HF_HOME", str(root / "cache" / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(root / "cache" / "torch"))
    return ProjectStorage(root=root)
