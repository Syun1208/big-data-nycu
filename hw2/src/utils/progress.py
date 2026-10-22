from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tqdm import tqdm

COUNT_BAR = (
    "{desc} ▕{bar}▏ {percentage:3.0f}% • {n_fmt}/{total_fmt} • {rate_fmt}"
    " • ⏳ {elapsed}<{remaining}{postfix}"
)


@dataclass(frozen=True, slots=True)
class Stage:
    emoji: str
    title: str
    colour: str


LOAD = Stage(emoji="📥", title="Load tables", colour="cyan")
FEATURES = Stage(emoji="🧩", title="Build features", colour="green")
TRAIN = Stage(emoji="🚀", title="Train models", colour="blue")
GRAPHS = Stage(emoji="📊", title="Render graphs", colour="magenta")


def create_progress(
    *,
    stage: Stage,
    total: int,
    unit: str = "it",
    iterable: Iterable | None = None,
) -> tqdm:
    return tqdm(
        iterable,
        total=total,
        unit=unit,
        desc=f"{stage.emoji} {stage.title}",
        colour=stage.colour,
        bar_format=COUNT_BAR,
        dynamic_ncols=True,
    )
