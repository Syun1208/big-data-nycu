from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

BOLD = "\033[1m"
GREEN = "\033[32m"
DIM = "\033[2m"
RESET = "\033[0m"


@dataclass(frozen=True, slots=True)
class Column:
    key: str
    title: str
    formatter: Callable[[object], str] = str
    align_right: bool = False


def percent(value: object) -> str:
    return "" if value is None or pd.isna(value) else f"{100 * float(value):.2f}"


def use_colour() -> bool:
    return not os.environ.get("NO_COLOR") and sys.stderr.isatty()


def box_table(
    *,
    frame: pd.DataFrame,
    columns: Sequence[Column],
    highlight: Sequence[bool] | None = None,
    dimmed: Sequence[bool] | None = None,
    title: str = "",
) -> str:
    cells = [[column.formatter(row[column.key]) for column in columns] for _, row in frame.iterrows()]
    widths = [max([len(column.title), *(len(row[index]) for row in cells)]) for index, column in enumerate(columns)]
    colour = use_colour()

    def line(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def render(values: Sequence[str]) -> str:
        parts = []
        for value, width, column in zip(values, widths, columns):
            parts.append(f" {value:>{width}} " if column.align_right else f" {value:<{width}} ")
        return "│" + "│".join(parts) + "│"

    lines = [title] if title else []
    lines.append(line("┌", "┬", "┐"))
    lines.append(render([column.title for column in columns]))
    lines.append(line("├", "┼", "┤"))
    for index, values in enumerate(cells):
        text = render(values)
        if colour and highlight is not None and highlight[index]:
            text = f"{BOLD}{GREEN}{text}{RESET}"
        elif colour and dimmed is not None and dimmed[index]:
            text = f"{DIM}{text}{RESET}"
        lines.append(text)
    lines.append(line("└", "┴", "┘"))
    return "\n".join(lines)
