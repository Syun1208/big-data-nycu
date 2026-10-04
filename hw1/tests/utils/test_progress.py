from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils.progress import Stage, create_progress

SEED = 42
DATASET_SIZE = 64
BATCH_SIZE = 8
STAGE = Stage(emoji="🚀", title="Train", colour="green")


def build_loader() -> DataLoader:
    return DataLoader(
        dataset=list(range(DATASET_SIZE)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )


def flatten(*, batches: object) -> list[int]:
    return [int(value) for batch in batches for value in batch]


def test_progress_iteration_keeps_notebook_shuffle_order() -> None:
    notebook_order = flatten(batches=tqdm(build_loader()))
    loader = build_loader()
    refactor_order = flatten(
        batches=create_progress(
            stage=STAGE,
            total=len(loader),
            unit="batch",
            iterable=loader,
        )
    )
    assert refactor_order == notebook_order
    assert sorted(refactor_order) == list(range(DATASET_SIZE))
