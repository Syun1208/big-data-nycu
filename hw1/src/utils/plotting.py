from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

ACADEMIC_PALETTE = ("#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B3", "#937860")
ACADEMIC_MARKERS = ("o", "s", "^", "D", "v", "P")
ACADEMIC_HATCHES = ("", "//", "..", "xx", "\\\\", "--")
ACADEMIC_LINESTYLES = ("-", "--", "-.", ":")
FIGURE_DPI = 300
FIGURE_SUFFIXES = (".pdf", ".png")
SINGLE_COLUMN_SIZE = (3.5, 2.4)
SPECTRUM_SIZE = (8.0, 3.2)
WIDE_SIZE = (5.0, 2.6)
BAR_EDGE_COLOUR = "#333333"
THRESHOLD_COLOUR = "#7F7F7F"
PERCENT_SCALE = 100.0


def apply_academic_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler(color=ACADEMIC_PALETTE, marker=ACADEMIC_MARKERS),
            "axes.grid": False,
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "hatch.linewidth": 0.6,
            "legend.frameon": False,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def save_figure(
    *,
    figure: plt.Figure,
    path: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in FIGURE_SUFFIXES:
        target = path.with_suffix(suffix)
        figure.savefig(target)
        written.append(target)
    plt.close(figure)
    return written


def plot_grouped_accuracy(
    *,
    accuracies: Mapping[str, Sequence[float | None]],
    group_labels: Sequence[str],
    series_colours: Mapping[str, str],
) -> plt.Figure:
    import matplotlib.pyplot as plt
    import numpy as np

    series_names = list(accuracies)
    positions = np.arange(len(group_labels))
    width = 0.8 / max(1, len(series_names))
    figure, axis = plt.subplots(figsize=WIDE_SIZE)

    for index, name in enumerate(series_names):
        values = [np.nan if value is None else PERCENT_SCALE * value for value in accuracies[name]]
        offsets = positions + (index - (len(series_names) - 1) / 2) * width
        bars = axis.bar(
            offsets,
            values,
            width=width,
            label=name,
            color=series_colours[name],
            edgecolor=BAR_EDGE_COLOUR,
            linewidth=0.6,
            hatch=ACADEMIC_HATCHES[index % len(ACADEMIC_HATCHES)],
        )
        axis.bar_label(
            bars,
            labels=["" if np.isnan(value) else f"{value:.1f}" for value in values],
            padding=1.5,
            fontsize=7,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(group_labels)
    axis.set_xlabel("Singular-value range (target rank = 16)")
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(bottom=0)
    axis.margins(y=0.15)
    axis.yaxis.grid(True)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper right",
        ncol=len(series_names),
    )
    return figure


def plot_singular_values(
    *,
    singular_values: Sequence[float],
    colour: str,
    label: str,
) -> plt.Figure:
    import matplotlib.pyplot as plt
    import numpy as np

    indices = np.arange(1, len(singular_values) + 1)
    figure, axis = plt.subplots(figsize=SPECTRUM_SIZE)
    axis.bar(
        indices,
        singular_values,
        width=1.0,
        color=colour,
        edgecolor=colour,
        linewidth=0.0,
        label=label,
    )
    axis.set_xlabel("Singular value index")
    axis.set_ylabel("Singular value magnitude")
    axis.set_xlim(0.5, len(singular_values) + 0.5)
    axis.legend(loc="upper right")
    return figure


def plot_cumulative_energy(
    *,
    cumulative_energy_percent: Sequence[float],
    threshold_percent: float,
    minimum_rank: int,
    colour: str,
    marker: str,
    label: str,
) -> plt.Figure:
    import matplotlib.pyplot as plt
    import numpy as np

    ranks = np.arange(1, len(cumulative_energy_percent) + 1)
    figure, axis = plt.subplots(figsize=SPECTRUM_SIZE)
    axis.plot(
        ranks,
        cumulative_energy_percent,
        color=colour,
        marker=marker,
        markevery=max(1, len(ranks) // 10),
        label=label,
    )
    axis.axhline(
        threshold_percent,
        color=THRESHOLD_COLOUR,
        linestyle="--",
        linewidth=0.8,
        label=f"{threshold_percent:.0f}% energy",
    )
    if minimum_rank <= len(ranks):
        axis.axvline(
            minimum_rank,
            color=THRESHOLD_COLOUR,
            linestyle=":",
            linewidth=0.8,
        )
        axis.annotate(
            f"r = {minimum_rank}",
            xy=(minimum_rank, threshold_percent),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("Target rank")
    axis.set_ylabel("Cumulative energy (%)")
    axis.set_xlim(1, len(ranks))
    axis.set_ylim(0, 100)
    axis.yaxis.grid(True)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right")
    return figure
