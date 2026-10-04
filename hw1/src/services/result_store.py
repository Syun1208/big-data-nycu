from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

FRESH_METHOD = "fresh"
PISSA_METHOD = "pissa"
BENCHMARK_STEM = "benchmark_results"


def build_run_tag(
    *,
    method: str,
    rank: int | None = None,
    component: str | None = None,
) -> str:
    if method == FRESH_METHOD:
        return "metamath_fresh"
    if method != PISSA_METHOD:
        raise ValueError(f"Unsupported method: {method}")
    return f"metamath_pissa_{component}_r{rank}"


class ResultStore:
    def __init__(
        self,
        *,
        root: Path,
        save_predictions: bool,
    ) -> None:
        self._root = root
        self._save_predictions = save_predictions
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    @property
    def benchmark_csv_path(self) -> Path:
        return self.root / f"{BENCHMARK_STEM}.csv"

    @property
    def benchmark_json_path(self) -> Path:
        return self.root / f"{BENCHMARK_STEM}.json"

    def summary_path(self, *, tag: str) -> Path:
        return self.root / f"{tag}_summary.json"

    def predictions_path(self, *, tag: str) -> Path:
        return self.root / f"{tag}_predictions.jsonl"

    def training_time_path(self, *, tag: str) -> Path:
        return self.root / f"{tag}_training_time.json"

    def training_history_path(self, *, tag: str) -> Path:
        return self.root / f"{tag}_training_history.csv"

    def adapter_path(self, *, tag: str) -> Path:
        return self.root / f"{tag}_adapter.pt"

    def full_model_path(self, *, tag: str) -> Path:
        return self.root / f"{tag}_full_model"

    def save_summary(
        self,
        *,
        summary: Mapping[str, Any],
        records: Iterable[Mapping[str, Any]] | None = None,
    ) -> Path:
        tag = summary["run_tag"]
        path = self.summary_path(tag=tag)
        _write_json(
            path=path,
            payload=summary,
        )
        if self._save_predictions and records is not None:
            _write_jsonl(
                path=self.predictions_path(tag=tag),
                records=records,
            )
        logger.info("💾 Summary saved: %s", path)
        return path

    def save_training_time_partial(self, *, summary: Mapping[str, Any]) -> Path:
        path = self.training_time_path(tag=summary["run_tag"])
        _write_json(
            path=path,
            payload=summary,
        )
        return path

    def save_training_history(
        self,
        *,
        tag: str,
        history: pd.DataFrame,
    ) -> Path:
        path = self.training_history_path(tag=tag)
        history.to_csv(path, index=False)
        logger.info("💾 Training history saved: %s", path)
        return path

    def save_master_results(self, *, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        import pandas as pd

        frame = pd.DataFrame(list(rows))
        frame.to_csv(self.benchmark_csv_path, index=False)
        _write_json(
            path=self.benchmark_json_path,
            payload=list(rows),
        )
        return frame


def _write_json(
    *,
    path: Path,
    payload: object,
) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def _write_jsonl(
    *,
    path: Path,
    records: Iterable[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
