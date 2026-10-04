from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)

TRAIN_SPLIT = "train"
TEST_SPLIT = "test"


class MetaMathTrainLoader:
    def __init__(
        self,
        *,
        dataset_id: str,
        limit: int | None,
    ) -> None:
        self._dataset_id = dataset_id
        self._limit = limit

    def load(self) -> Dataset:
        from datasets import load_dataset

        split = TRAIN_SPLIT if self._limit is None else f"{TRAIN_SPLIT}[:{self._limit}]"
        logger.info("📥 Loading %s split=%s", self._dataset_id, split)
        train = load_dataset(
            self._dataset_id,
            split=split,
        )
        return train.map(
            _format_metamath_row,
            remove_columns=train.column_names,
            desc="Formatting MetaMathQA for PiSSA training",
        )


class PiSSAEvalLoader:
    def __init__(
        self,
        *,
        dataset_id: str,
        data_dir: str,
        tasks: Sequence[str],
    ) -> None:
        self._dataset_id = dataset_id
        self._data_dir = data_dir
        self._tasks = tuple(tasks)

    def load(self) -> dict[str, Dataset]:
        from datasets import load_dataset

        logger.info("📥 Loading %s/%s split=%s", self._dataset_id, self._data_dir, TEST_SPLIT)
        raw_test = load_dataset(
            self._dataset_id,
            data_dir=self._data_dir,
            split=TEST_SPLIT,
        )
        normalized_types = [str(task_type).lower() for task_type in raw_test["type"]]

        eval_sets = {}
        for task in self._tasks:
            keep_indices = [index for index, task_type in enumerate(normalized_types) if task_type == task]
            subset = raw_test.select(keep_indices)
            if len(subset) == 0:
                raise RuntimeError(f"No {task!r} rows found in {self._dataset_id}/{self._data_dir} test split.")
            eval_sets[task] = subset.map(
                _format_eval_row,
                remove_columns=subset.column_names,
                desc=f"Formatting PiSSA {task} evaluation set",
            )
        return eval_sets


def _format_metamath_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "instruction": str(row["query"]).strip(),
        "output": str(row["response"]).strip(),
        "source_type": str(row.get("type", "")),
    }


def _format_eval_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "instruction": str(row["instruction"]),
        "gold_answer": str(row["output"]),
        "type": str(row["type"]).lower(),
    }
