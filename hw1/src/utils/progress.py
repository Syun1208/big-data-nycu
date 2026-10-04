from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tqdm.auto import tqdm

COUNT_BAR = (
    "{desc} ▕{bar}▏ {percentage:3.0f}% • {n_fmt}/{total_fmt} • {rate_fmt}"
    " • ⏳ {elapsed}<{remaining}{postfix}"
)
TIME_BAR = "{desc} ▕{bar}▏ {percentage:3.0f}% • ⏳ {elapsed}<{remaining}{postfix}"


@dataclass(frozen=True, slots=True)
class Stage:
    emoji: str
    title: str
    colour: str


LOAD = Stage(emoji="📥", title="Load", colour="cyan")
PROCESS = Stage(emoji="🧩", title="Process", colour="green")
SAVE = Stage(emoji="💾", title="Save", colour="magenta")


def create_progress(
    *,
    stage: Stage,
    total: float | None,
    unit: str = "it",
    show_counts: bool = True,
    iterable: Iterable | None = None,
) -> tqdm:
    layout = COUNT_BAR if show_counts else TIME_BAR
    return tqdm(
        iterable,
        total=total,
        unit=unit,
        desc=f"{stage.emoji} {stage.title}",
        colour=stage.colour,
        bar_format=layout if total else None,
        dynamic_ncols=True,
    )
