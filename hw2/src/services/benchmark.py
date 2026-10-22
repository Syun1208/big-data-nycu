from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.data.classes.settings import EVENTS
from src.services.comparison import markdown_table
from src.services.evaluation import USER_COLUMN, run_official_evaluator
from src.services.prediction import OUTPUT_FILE
from src.services.scoring import apply_thresholds
from src.services.training import HOLDOUT_FILE, METRICS_FILE, THRESHOLDS_FILE
from src.utils.table import Column, box_table, percent

logger = logging.getLogger(__name__)

HOLDOUT_OUTPUT_FILE = "holdout_output.csv"
HOLDOUT_TRUTH_FILE = "holdout_truth.csv"
BENCHMARK_FILE = "benchmark.csv"
BENCHMARK_MARKDOWN = "benchmark.md"
METHOD_NAMES = {
    "xgboost": "XGBoost",
    "rdl_gnn": "Hetero GNN",
    "rdl_gnn_hybrid": "Hybrid GNN",
    "rdl_gnn_hybrid_gat": "Hybrid GAT",
    "rdl_gnn_hybrid_transformer": "Hybrid Graph Transformer",
    "tabpfn_v2": "TabPFN v2",
    "ensemble": "Ensemble",
}
BASELINE_NAMES = {
    "last_7d": "Baseline: active last 7 days",
    "last_28d": "Baseline: active last 28 days",
    "push_only": "Baseline: push only",
}


def best_per_method(*, table: pd.DataFrame, submission_run: str | None) -> pd.DataFrame:
    table = table.assign(submitted=table["run"].eq(submission_run))
    return table.sort_values(["submitted", "holdout_score"], ascending=False).drop_duplicates("method", keep="first")


def baseline_rows(*, run_dir: Path) -> list[dict]:
    metrics = json.loads((run_dir / METRICS_FILE).read_text(encoding="utf-8"))
    rows = []
    for name, values in metrics.get("holdout_baselines", {}).items():
        row = {
            "method": BASELINE_NAMES.get(name, name),
            "holdout_score": values["score"],
            "holdout_macro_f1": values["macro_f1"],
            "holdout_micro_f1": values["micro_f1"],
            "is_baseline": True,
        }
        row.update({f"f1_{event}": values["per_event_f1"][event] for event in EVENTS})
        rows.append(row)
    return rows


def find_thresholds(*, run_dir: Path, model_root: Path) -> Path | None:
    for folder in (run_dir, model_root / run_dir.name):
        if (folder / THRESHOLDS_FILE).is_file():
            return folder / THRESHOLDS_FILE
    return None


def write_holdout_files(
    *,
    run_dir: Path,
    thresholds_path: Path,
) -> tuple[Path, Path]:
    frame = pd.read_parquet(run_dir / HOLDOUT_FILE)
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    predictions = apply_thresholds(
        probabilities=frame[[f"proba_{event}" for event in EVENTS]].to_numpy(),
        thresholds=[thresholds[event] for event in EVENTS],
    )
    users = frame["github_user_id"].astype("int64").to_numpy()
    output = pd.DataFrame(predictions, columns=list(EVENTS)).astype("int64")
    output.insert(0, USER_COLUMN, users)
    truth = frame[[f"label_{event}" for event in EVENTS]].astype("int64")
    truth.columns = list(EVENTS)
    truth.insert(0, USER_COLUMN, users)
    output_path = run_dir / HOLDOUT_OUTPUT_FILE
    truth_path = run_dir / HOLDOUT_TRUTH_FILE
    output.to_csv(output_path, index=False)
    truth.to_csv(truth_path, index=False)
    return output_path, truth_path


class OutputBenchmark:
    def __init__(
        self,
        *,
        evaluator: Path,
        model_root: Path,
        submission_run: str | None,
    ) -> None:
        self._evaluator = evaluator
        self._model_root = model_root
        self._submission_run = submission_run

    def run(
        self,
        *,
        run_dirs: Sequence[Path],
        output_dir: Path,
    ) -> pd.DataFrame:
        rows = []
        for run_dir in run_dirs:
            row = self._score_run(run_dir=run_dir)
            if row is not None:
                rows.append(row)
        table = pd.DataFrame(rows).sort_values("holdout_score", ascending=False).reset_index(drop=True)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        table.to_csv(output_dir / BENCHMARK_FILE, index=False)
        summary = ["method", "run", "holdout_score", "holdout_macro_f1", "holdout_micro_f1", "calibration_score", "test_format"]
        events = ["method", *[f"f1_{event}" for event in EVENTS]]
        rates = ["method", *[f"test_rate_{event}" for event in EVENTS]]
        markdown = "\n\n".join(markdown_table(frame=table[columns]) for columns in (summary, events, rates))
        (output_dir / BENCHMARK_MARKDOWN).write_text(markdown + "\n", encoding="utf-8")
        logger.info(
            "📊 Validation week scores from the TA evaluate.py (full table: %s)\n%s",
            output_dir / BENCHMARK_FILE,
            self._pretty(table=table, baselines=baseline_rows(run_dir=run_dirs[0]) if run_dirs else []),
        )
        return table

    def _pretty(
        self,
        *,
        table: pd.DataFrame,
        baselines: list[dict],
    ) -> str:
        shown = best_per_method(table=table, submission_run=self._submission_run)
        names = shown["method"].map(lambda name: METHOD_NAMES.get(name, name))
        shown = shown.assign(method=names.where(~shown["submitted"], names + " (submitted)"), is_baseline=False)
        shown = pd.concat([shown, pd.DataFrame(baselines)], ignore_index=True)
        shown = shown.sort_values("holdout_score", ascending=False).reset_index(drop=True)
        model_rows = ~shown["is_baseline"].astype(bool)
        shown["rank"] = ""
        shown.loc[model_rows, "rank"] = [str(index + 1) for index in range(int(model_rows.sum()))]
        highlight = shown["submitted"].fillna(False).astype(bool).tolist()
        dimmed = shown["is_baseline"].astype(bool).tolist()
        return box_table(
            frame=shown,
            columns=[
                Column("rank", "#", align_right=True),
                Column("method", "Method"),
                Column("holdout_macro_f1", "Macro-F1 %", percent, True),
                Column("holdout_micro_f1", "Micro-F1 %", percent, True),
                Column("holdout_score", "Score %", percent, True),
            ],
            highlight=highlight,
            dimmed=dimmed,
            title="Best run per method, validation week 10/07, Score = (macro-F1 + micro-F1) / 2",
        )

    def _score_run(self, *, run_dir: Path) -> dict | None:
        thresholds_path = find_thresholds(run_dir=run_dir, model_root=self._model_root)
        if not (run_dir / HOLDOUT_FILE).is_file() or thresholds_path is None:
            logger.warning("Skipped %s: no holdout predictions or thresholds", run_dir.name)
            return None
        output_path, truth_path = write_holdout_files(
            run_dir=run_dir,
            thresholds_path=thresholds_path,
        )
        official = run_official_evaluator(
            evaluator=self._evaluator,
            prediction=output_path,
            truth=truth_path,
            test_input=None,
        )
        if official.returncode != 0:
            logger.error("evaluate.py rejected %s: %s", output_path, official.output)
            return None
        metrics = json.loads((run_dir / METRICS_FILE).read_text(encoding="utf-8")) if (run_dir / METRICS_FILE).is_file() else {}
        row = {
            "method": metrics.get("model", run_dir.name),
            "run": run_dir.name,
            "holdout_score": official.score_line(name="Score"),
            "holdout_macro_f1": official.score_line(name="macro-F1"),
            "holdout_micro_f1": official.score_line(name="micro-F1"),
            "calibration_score": metrics.get("calibration", {}).get("score"),
        }
        truth = pd.read_csv(truth_path)
        predicted = pd.read_csv(output_path)
        for event in EVENTS:
            tp = int(((predicted[event] == 1) & (truth[event] == 1)).sum())
            fp = int(((predicted[event] == 1) & (truth[event] == 0)).sum())
            fn = int(((predicted[event] == 0) & (truth[event] == 1)).sum())
            row[f"f1_{event}"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
        row.update(self._test_output_stats(run_dir=run_dir))
        logger.info("%-34s holdout score %.4f", row["method"], row["holdout_score"])
        return row

    def _test_output_stats(self, *, run_dir: Path) -> dict:
        path = run_dir / OUTPUT_FILE
        if not path.is_file():
            return {"test_format": "missing"}
        test_input = self._evaluator.parent / "test_input.csv"
        official = run_official_evaluator(
            evaluator=self._evaluator,
            prediction=path,
            truth=None,
            test_input=test_input if test_input.is_file() else None,
        )
        output = pd.read_csv(path)
        stats = {"test_format": "ok" if official.returncode == 0 else "ERROR", "test_users": len(output)}
        stats.update({f"test_rate_{event}": float(output[event].mean()) for event in EVENTS})
        return stats
