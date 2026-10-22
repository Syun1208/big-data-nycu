from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.data.classes.settings import EVENTS
from src.services.prediction import OUTPUT_FILE
from src.services.training import METRICS_FILE

logger = logging.getLogger(__name__)

COMPARISON_FILE = "comparison.csv"
COMPARISON_MARKDOWN = "comparison.md"


def metrics_row(*, run_dir: Path) -> dict:
    metrics = json.loads((run_dir / METRICS_FILE).read_text(encoding="utf-8"))
    holdout = metrics["holdout"]
    row = {
        "run": run_dir.name,
        "model": metrics.get("model", metrics["experiment"]),
        "holdout_score": holdout["score"],
        "holdout_macro_f1": holdout["macro_f1"],
        "holdout_micro_f1": holdout["micro_f1"],
        "calibration_score": metrics["calibration"]["score"],
        "train_minutes": metrics.get("train_minutes"),
    }
    row.update({f"f1_{event}": holdout["per_event_f1"][event] for event in EVENTS})
    row["has_output"] = (run_dir / OUTPUT_FILE).is_file()
    return row


def baseline_rows(*, run_dir: Path) -> list[dict]:
    metrics = json.loads((run_dir / METRICS_FILE).read_text(encoding="utf-8"))
    rows = []
    for name, values in metrics["holdout_baselines"].items():
        row = {
            "run": "baseline",
            "model": name,
            "holdout_score": values["score"],
            "holdout_macro_f1": values["macro_f1"],
            "holdout_micro_f1": values["micro_f1"],
        }
        row.update({f"f1_{event}": values["per_event_f1"][event] for event in EVENTS})
        rows.append(row)
    return rows


def markdown_table(*, frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if isinstance(value, float):
            return "" if pd.isna(value) else f"{value:.4f}"
        return str(value)

    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


class RunComparator:
    def __init__(self, *, output_dir: Path) -> None:
        self._output_dir = output_dir

    def compare(
        self,
        *,
        run_dirs: Sequence[Path],
        submission_dir: Path | None,
    ) -> pd.DataFrame:
        available = [run_dir for run_dir in run_dirs if (run_dir / METRICS_FILE).is_file()]
        for run_dir in run_dirs:
            if run_dir not in available:
                logger.warning("No %s in %s. Skipped.", METRICS_FILE, run_dir)
        if not available:
            raise FileNotFoundError("No trained run to compare")
        table = pd.DataFrame([metrics_row(run_dir=run_dir) for run_dir in available])
        table = pd.concat([table, pd.DataFrame(baseline_rows(run_dir=available[0]))], ignore_index=True)
        table = table.sort_values("holdout_score", ascending=False).reset_index(drop=True)
        self._output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        table.to_csv(self._output_dir / COMPARISON_FILE, index=False)
        columns = ["model", "run", "holdout_score", "holdout_macro_f1", "holdout_micro_f1", "calibration_score"]
        markdown = markdown_table(frame=table[columns])
        (self._output_dir / COMPARISON_MARKDOWN).write_text(markdown + "\n", encoding="utf-8")
        logger.info("📊 Holdout comparison (%s):\n%s", self._output_dir / COMPARISON_FILE, markdown)
        if submission_dir is not None:
            self._copy_best(
                table=table,
                run_dirs=available,
                submission_dir=submission_dir,
            )
        return table

    @staticmethod
    def _copy_best(
        *,
        table: pd.DataFrame,
        run_dirs: Sequence[Path],
        submission_dir: Path,
    ) -> None:
        by_name = {run_dir.name: run_dir for run_dir in run_dirs}
        ranked = table[table["run"].isin(by_name) & table["has_output"].fillna(False).astype(bool)]
        ranked = ranked.sort_values("calibration_score", ascending=False)
        if ranked.empty:
            logger.warning("No run has %s yet. Run predict first.", OUTPUT_FILE)
            return
        best = ranked.iloc[0]
        submission_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.copy2(by_name[best["run"]] / OUTPUT_FILE, submission_dir / OUTPUT_FILE)
        (submission_dir / "best_run.txt").write_text(f"{best['run']}\n", encoding="utf-8")
        logger.info(
            "🏆 Best model by calibration score: %s (calibration %.4f, holdout %.4f). Copied to %s",
            best["run"],
            best["calibration_score"],
            best["holdout_score"],
            submission_dir / OUTPUT_FILE,
        )
