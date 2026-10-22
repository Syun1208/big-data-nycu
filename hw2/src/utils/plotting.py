from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

FIGURE_DPI = 300
SERIF_FONTS = ("Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif")
INK = "#000000"
CELL_LOW = "#FFFFFF"
CELL_HIGH = "#08519C"
CELL_BORDER = "#C6D2E3"
NEGATIVE_POINT = "#1F3F77"
POSITIVE_POINT = "#D62728"
FULL_WIDTH_INCHES = 7.0
PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def apply_academic_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": list(SERIF_FONTS),
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.grid": False,
            "legend.frameon": False,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def panel_title(*, index: int, text: str) -> str:
    return f"({PANEL_LETTERS[index]}) {text}"


def save_png(
    *,
    figure: plt.Figure,
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    target = path.with_suffix(".png")
    figure.savefig(target, dpi=FIGURE_DPI)
    plt.close(figure)
    return target
