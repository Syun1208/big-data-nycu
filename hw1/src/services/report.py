from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.data.classes.settings import COMPONENT_LABELS, DEFAULT_COMPONENT
from src.services.result_store import FRESH_METHOD, PISSA_METHOD
from src.utils.plotting import ACADEMIC_PALETTE, apply_academic_style, plot_grouped_accuracy, save_figure

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

DATASET_COLUMNS = {
    "GSM8K": "gsm8k_accuracy",
    "MATH": "math_accuracy",
}
DATASET_COLOURS = {
    "GSM8K": ACADEMIC_PALETTE[0],
    "MATH": ACADEMIC_PALETTE[1],
}
ORIGINAL_COLUMN = "Original LLM"
COMPONENT_RANK = 16
PERCENT_SCALE = 100.0
PERCENT_DECIMALS = 2
MINUTE_DECIMALS = 2


@dataclass(frozen=True, slots=True)
class ReportTables:
    accuracy: pd.DataFrame | None
    training_time: pd.DataFrame | None
    component_accuracy: pd.DataFrame | None


def load_successful_results(*, paths: Sequence[Path]) -> pd.DataFrame:
    import pandas as pd

    frames = [pd.read_csv(path) for path in paths]
    results = pd.concat(frames, ignore_index=True)
    if "status" in results.columns:
        results = results[results["status"].eq("ok")]
    return results.drop_duplicates(
        subset="run_tag",
        keep="last",
    ).reset_index(drop=True)


def fine_tuned_column(*, rank: int) -> str:
    return f"Fine-Tuned LLM (RANK = {rank})"


def build_accuracy_table(*, results: pd.DataFrame) -> pd.DataFrame | None:
    import pandas as pd

    default_runs = _default_component_runs(results=results)
    fresh = results[results["method"].eq(FRESH_METHOD)]
    if default_runs.empty and fresh.empty:
        return None

    table = pd.DataFrame(index=list(DATASET_COLUMNS))
    table.index.name = "Accuracy (%)"
    if not fresh.empty:
        table[ORIGINAL_COLUMN] = [_percent(value=fresh.iloc[-1][column]) for column in DATASET_COLUMNS.values()]
    for _, row in default_runs.iterrows():
        table[fine_tuned_column(rank=int(row["rank"]))] = [
            _percent(value=row[column]) for column in DATASET_COLUMNS.values()
        ]
    return table


def build_training_time_table(*, results: pd.DataFrame) -> pd.DataFrame | None:
    import pandas as pd

    default_runs = _default_component_runs(results=results)
    if default_runs.empty or "train_minutes" not in default_runs.columns:
        return None

    table = pd.DataFrame(index=["Training Time (min.)"])
    for _, row in default_runs.iterrows():
        table[fine_tuned_column(rank=int(row["rank"]))] = [round(float(row["train_minutes"]), MINUTE_DECIMALS)]
    return table


def build_component_accuracy_table(*, results: pd.DataFrame) -> pd.DataFrame | None:
    import pandas as pd

    pissa = results[results["method"].eq(PISSA_METHOD) & results["rank"].eq(COMPONENT_RANK)]
    ordered = [component for component in COMPONENT_LABELS if component in set(pissa["component"])]
    if len(ordered) < 2:
        return None

    rows = {}
    for component in ordered:
        row = pissa[pissa["component"].eq(component)].iloc[-1]
        rows[COMPONENT_LABELS[component]] = [_percent(value=row[column]) for column in DATASET_COLUMNS.values()]
    table = pd.DataFrame.from_dict(
        rows,
        orient="index",
        columns=list(DATASET_COLUMNS),
    )
    table.index.name = "Singular-value range (rank 16)"
    return table


def build_report_tables(*, results: pd.DataFrame) -> ReportTables:
    return ReportTables(
        accuracy=build_accuracy_table(results=results),
        training_time=build_training_time_table(results=results),
        component_accuracy=build_component_accuracy_table(results=results),
    )


def _default_component_runs(*, results: pd.DataFrame) -> pd.DataFrame:
    runs = results[results["method"].eq(PISSA_METHOD) & results["component"].eq(DEFAULT_COMPONENT)]
    return runs.sort_values("rank")


def _percent(*, value: float) -> float:
    return round(PERCENT_SCALE * float(value), PERCENT_DECIMALS)


class ReportWriter:
    def __init__(self, *, output_dir: Path) -> None:
        self._output_dir = output_dir

    @property
    def markdown_path(self) -> Path:
        return self._output_dir / "report_tables.md"

    @property
    def component_figure_path(self) -> Path:
        return self._output_dir / "figures" / "q3_component_accuracy"

    def table_path(self, *, name: str) -> Path:
        return self._output_dir / f"{name}.csv"

    def write(self, *, tables: ReportTables) -> list[Path]:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        sections = (
            ("q1_accuracy", "Q1: Accuracy of original and fine-tuned Llama-3.2-1B", tables.accuracy),
            ("q2_training_time", "Q2: Training time by target rank", tables.training_time),
            ("q3_component_accuracy", "Q3: Accuracy by singular-value range", tables.component_accuracy),
        )
        written = []
        markdown = []
        for name, title, table in sections:
            if table is None:
                logger.warning("Skipping %s: required runs not found in results", name)
                continue
            path = self.table_path(name=name)
            table.to_csv(path)
            written.append(path)
            markdown.append(f"## {title}\n\n{table.to_markdown()}\n")
            logger.info("💾 %s: %s", name, path)

        if markdown:
            self.markdown_path.write_text(
                "\n".join(markdown),
                encoding="utf-8",
            )
            written.append(self.markdown_path)

        if tables.component_accuracy is not None:
            written += self._write_component_figure(table=tables.component_accuracy)
        return written

    def _write_component_figure(self, *, table: pd.DataFrame) -> list[Path]:
        apply_academic_style()
        figure = plot_grouped_accuracy(
            accuracies={
                dataset: [value / PERCENT_SCALE for value in table[dataset].tolist()] for dataset in DATASET_COLUMNS
            },
            group_labels=list(table.index),
            series_colours=DATASET_COLOURS,
        )
        paths = save_figure(
            figure=figure,
            path=self.component_figure_path,
        )
        logger.info("📊 Q3 figure: %s", paths[0])
        return paths
