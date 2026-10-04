from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.services.report import ReportWriter, build_report_tables, load_successful_results


def build_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"run_tag": "metamath_fresh", "method": "fresh", "rank": None, "component": None,
             "gsm8k_accuracy": 0.05, "math_accuracy": 0.02, "status": "ok"},
            {"run_tag": "metamath_pissa_default_r64", "method": "pissa", "rank": 64, "component": "default",
             "gsm8k_accuracy": 0.30, "math_accuracy": 0.06, "train_minutes": 22.5, "status": "ok"},
            {"run_tag": "metamath_pissa_default_r16", "method": "pissa", "rank": 16, "component": "default",
             "gsm8k_accuracy": 0.25, "math_accuracy": 0.05, "train_minutes": 20.0, "status": "ok"},
            {"run_tag": "metamath_pissa_p25_r16", "method": "pissa", "rank": 16, "component": "p25",
             "gsm8k_accuracy": 0.10, "math_accuracy": 0.03, "train_minutes": 20.1, "status": "ok"},
            {"run_tag": "metamath_pissa_bottom_r16", "method": "pissa", "rank": 16, "component": "bottom",
             "gsm8k_accuracy": 0.08, "math_accuracy": 0.02, "train_minutes": 20.2, "status": "ok"},
            {"run_tag": "metamath_pissa_p50_r16", "method": "pissa", "rank": 16, "component": "p50",
             "status": "oom"},
        ]
    )


def test_tables_follow_assignment_layout(tmp_path: Path) -> None:
    csv_path = tmp_path / "benchmark_results.csv"
    build_results().to_csv(csv_path, index=False)
    tables = build_report_tables(results=load_successful_results(paths=[csv_path]))

    assert list(tables.accuracy.columns) == [
        "Original LLM",
        "Fine-Tuned LLM (RANK = 16)",
        "Fine-Tuned LLM (RANK = 64)",
    ]
    assert tables.accuracy.loc["GSM8K", "Fine-Tuned LLM (RANK = 64)"] == 30.0
    assert tables.training_time.loc["Training Time (min.)", "Fine-Tuned LLM (RANK = 16)"] == 20.0
    assert list(tables.component_accuracy.index) == ["Top-16", "25% position", "Bottom-16"]

    written = ReportWriter(output_dir=tmp_path / "report").write(tables=tables)
    names = sorted(path.name for path in written)
    assert "q3_component_accuracy.pdf" in names
    assert "report_tables.md" in names
