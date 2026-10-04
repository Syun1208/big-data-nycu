from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
PISSA_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


class SupervisedExampleTokenizer:
    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __call__(self, example: Mapping[str, str]) -> dict[str, Any]:
        source = PISSA_PROMPT.format(instruction=example["instruction"])
        target = f"{example['output']}\n{self._tokenizer.eos_token}"

        input_ids = self._encode(text=source + target)
        source_ids = self._encode(text=source)

        labels = input_ids.copy()
        source_length = min(len(source_ids), len(labels))
        labels[:source_length] = [IGNORE_INDEX] * source_length

        return {
            "input_ids": input_ids,
            "labels": labels,
            "supervised_tokens": sum(label != IGNORE_INDEX for label in labels),
        }

    def tokenize_dataset(self, *, dataset: Dataset) -> Dataset:
        tokenized = dataset.map(
            self,
            remove_columns=dataset.column_names,
            desc="Tokenizing MetaMathQA",
        )
        before = len(tokenized)
        tokenized = tokenized.filter(
            _has_supervised_tokens,
            desc="Removing rows with no supervised tokens after truncation",
        )
        logger.info("🧩 MetaMathQA train rows: %d, usable: %d", before, len(tokenized))
        return tokenized

    def _encode(self, *, text: str) -> list[int]:
        return self._tokenizer(
            text,
            max_length=self._max_length,
            truncation=True,
            add_special_tokens=True,
        )["input_ids"]


@dataclass(frozen=True, slots=True)
class ResponseOnlyCollator:
    pad_token_id: int

    def __call__(self, features: Sequence[Mapping[str, list[int]]]) -> dict[str, torch.Tensor]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids, labels, attention_mask = [], [], []

        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            labels.append(feature["labels"] + [IGNORE_INDEX] * padding)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def _has_supervised_tokens(row: Mapping[str, Any]) -> bool:
    return row["supervised_tokens"] > 0
