from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.classes.settings import EVENTS
from src.services.scoring import apply_thresholds, f1_from_counts
from src.services.training import EMBEDDING_FILE, HOLDOUT_FILE, METRICS_FILE, THRESHOLDS_FILE, write_json
from src.utils.plotting import (
    CELL_BORDER,
    CELL_HIGH,
    CELL_LOW,
    FULL_WIDTH_INCHES,
    INK,
    NEGATIVE_POINT,
    POSITIVE_POINT,
    apply_academic_style,
    panel_title,
    save_png,
)

logger = logging.getLogger(__name__)

CONFUSION_JSON = "confusion_matrix.json"
CONFUSION_FIGURE = "confusion_matrix.png"
TSNE_FIGURE = "tsne.png"
CLASS_LABELS = ("Neg.", "Pos.")
WHITE_TEXT_SHARE = 0.55
CONFUSION_SIZE = (FULL_WIDTH_INCHES, 4.6)
TSNE_SIZE = (FULL_WIDTH_INCHES, 5.0)
NEGATIVE_LABEL = "No event in the next week"
POSITIVE_LABEL = "Event in the next week"


@dataclass(frozen=True, slots=True)
class DiagnosticSettings:
    tsne_max_points: int = 5000
    tsne_perplexity: float = 30.0
    tsne_jobs: int = 8
    seed: int = 42


def read_run_thresholds(*, run_dir: Path) -> list[float]:
    path = run_dir / THRESHOLDS_FILE
    values = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else json.loads((run_dir / METRICS_FILE).read_text(encoding="utf-8"))["thresholds"]
    return [float(values[event]) for event in EVENTS]


def confusion_table(
    *,
    predictions: np.ndarray,
    truth: np.ndarray,
) -> pd.DataFrame:
    predicted, actual = predictions == 1, truth == 1
    table = pd.DataFrame(
        {
            "tn": (~predicted & ~actual).sum(axis=0),
            "fp": (predicted & ~actual).sum(axis=0),
            "fn": (~predicted & actual).sum(axis=0),
            "tp": (predicted & actual).sum(axis=0),
        },
        index=list(EVENTS),
    )
    table["f1"] = f1_from_counts(tp=table["tp"], fp=table["fp"], fn=table["fn"])
    return table


def plot_confusion(
    *,
    table: pd.DataFrame,
    path: Path,
) -> Path:
    apply_academic_style()
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    colormap = LinearSegmentedColormap.from_list("white_blue", [CELL_LOW, CELL_HIGH])
    figure, grid = plt.subplots(2, 3, figsize=CONFUSION_SIZE, constrained_layout=True)
    image = None
    for index, (axes, event) in enumerate(zip(grid.flat, EVENTS)):
        row = table.loc[event]
        counts = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype="float64")
        shares = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
        image = axes.imshow(shares, cmap=colormap, vmin=0.0, vmax=1.0)
        for (y, x), count in np.ndenumerate(counts):
            color = "#FFFFFF" if shares[y, x] > WHITE_TEXT_SHARE else INK
            axes.text(x, y, f"{int(count):,}\n({shares[y, x]:.1%})", ha="center", va="center", fontsize=8, color=color)
        axes.set_xticks([0, 1], labels=CLASS_LABELS)
        axes.set_yticks([0, 1], labels=CLASS_LABELS)
        axes.set_xticks([0.5], minor=True)
        axes.set_yticks([0.5], minor=True)
        axes.grid(which="minor", color=CELL_BORDER, linewidth=0.8)
        axes.tick_params(which="minor", length=0)
        axes.tick_params(which="major", length=0)
        for spine in axes.spines.values():
            spine.set_color(CELL_BORDER)
            spine.set_linewidth(0.8)
        axes.set_title(panel_title(index=index, text=f"{event}\nF1 = {row['f1']:.3f}"))
        if index >= 3:
            axes.set_xlabel("Predicted label")
        if index % 3 == 0:
            axes.set_ylabel("True label")
    colorbar = figure.colorbar(image, ax=grid, fraction=0.025, pad=0.02)
    colorbar.set_label("Share of the true class")
    colorbar.outline.set_linewidth(0.6)
    return save_png(figure=figure, path=path)


def tsne_points(
    *,
    embeddings: np.ndarray,
    settings: DiagnosticSettings,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.manifold import TSNE

    rows = np.arange(len(embeddings))
    if len(rows) > settings.tsne_max_points:
        rows = np.sort(np.random.default_rng(settings.seed).choice(rows, size=settings.tsne_max_points, replace=False))
    points = TSNE(
        n_components=2,
        perplexity=settings.tsne_perplexity,
        init="pca",
        random_state=settings.seed,
        n_jobs=settings.tsne_jobs,
    ).fit_transform(embeddings[rows])
    return rows, points


def plot_tsne(
    *,
    points: np.ndarray,
    labels: np.ndarray,
    path: Path,
) -> Path:
    apply_academic_style()
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    figure, grid = plt.subplots(2, 3, figsize=TSNE_SIZE, constrained_layout=True)
    for index, (axes, event) in enumerate(zip(grid.flat, EVENTS)):
        positive = labels[:, index] == 1
        axes.scatter(points[~positive, 0], points[~positive, 1], s=2, color=NEGATIVE_POINT, alpha=0.5, linewidths=0, rasterized=True)
        axes.scatter(points[positive, 0], points[positive, 1], s=3, color=POSITIVE_POINT, linewidths=0, rasterized=True)
        axes.set_title(panel_title(index=index, text=f"{event}\n{positive.mean():.1%} positive"))
        axes.set_aspect("equal", adjustable="datalim")
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_linewidth(0.6)
        if index >= 3:
            axes.set_xlabel("t-SNE 1")
        if index % 3 == 0:
            axes.set_ylabel("t-SNE 2")
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=4, color=NEGATIVE_POINT, label=NEGATIVE_LABEL),
        Line2D([], [], marker="o", linestyle="", markersize=4, color=POSITIVE_POINT, label=POSITIVE_LABEL),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=2)
    return save_png(figure=figure, path=path)


class RunDiagnostics:
    def __init__(self, *, settings: DiagnosticSettings) -> None:
        self._settings = settings

    def run(self, *, run_dirs: Sequence[Path]) -> None:
        for run_dir in run_dirs:
            if not (run_dir / HOLDOUT_FILE).is_file():
                logger.warning("No %s in %s, skipped", HOLDOUT_FILE, run_dir)
                continue
            self._diagnose(run_dir=run_dir)

    def _diagnose(self, *, run_dir: Path) -> None:
        frame = pd.read_parquet(run_dir / HOLDOUT_FILE)
        probabilities = frame[[f"proba_{event}" for event in EVENTS]].to_numpy()
        truth = frame[[f"label_{event}" for event in EVENTS]].to_numpy()
        predictions = apply_thresholds(
            probabilities=probabilities,
            thresholds=read_run_thresholds(run_dir=run_dir),
        )
        table = confusion_table(predictions=predictions, truth=truth)
        write_json(path=run_dir / CONFUSION_JSON, payload=table.to_dict(orient="index"))
        plot_confusion(
            table=table,
            path=run_dir / CONFUSION_FIGURE,
        )
        logger.info("🧮 %s: confusion matrices in %s", run_dir.name, run_dir / CONFUSION_FIGURE)
        embedding_path = run_dir / EMBEDDING_FILE
        if not embedding_path.is_file():
            return
        embeddings = np.load(embedding_path)
        if len(embeddings) != len(truth):
            raise ValueError(f"{embedding_path} has {len(embeddings)} rows, {HOLDOUT_FILE} has {len(truth)}")
        rows, points = tsne_points(
            embeddings=embeddings,
            settings=self._settings,
        )
        plot_tsne(
            points=points,
            labels=truth[rows],
            path=run_dir / TSNE_FIGURE,
        )
        logger.info("🗺️ %s: t-SNE of %d users in %s", run_dir.name, len(rows), run_dir / TSNE_FIGURE)
