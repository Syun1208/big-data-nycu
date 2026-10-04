from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from src.data.classes.records import TaskEvaluation
from src.interface.answer_checker import AnswerChecker
from src.utils.cuda_memory import time_cuda_call
from src.utils.progress import Stage, create_progress

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

EVALUATE = Stage(emoji="🔎", title="Eval", colour="yellow")
LEFT_SIDE = "left"


class GenerationEvaluator:
    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        checkers: Mapping[str, AnswerChecker],
        device: torch.device,
        max_length: int,
        max_new_tokens: int,
        batch_size: int,
    ) -> None:
        self._tokenizer = tokenizer
        self._checkers = dict(checkers)
        self._device = device
        self._max_length = max_length
        self._max_new_tokens = max_new_tokens
        self._batch_size = batch_size

    def evaluate_all(
        self,
        *,
        model: PreTrainedModel,
        datasets: Mapping[str, Dataset],
        max_examples: int | None,
    ) -> dict[str, TaskEvaluation]:
        return {
            task_name: self.evaluate(
                model=model,
                dataset=dataset,
                task_name=task_name,
                max_examples=max_examples,
            )
            for task_name, dataset in datasets.items()
        }

    @torch.no_grad()
    def evaluate(
        self,
        *,
        model: PreTrainedModel,
        dataset: Dataset,
        task_name: str,
        max_examples: int | None,
    ) -> TaskEvaluation:
        if task_name not in self._checkers:
            raise ValueError(f"Unsupported evaluation task: {task_name}")

        model.eval()
        model.config.use_cache = True
        total = len(dataset) if max_examples is None else min(len(dataset), max_examples)

        records, eval_seconds = time_cuda_call(
            action=lambda: self._generate_and_check(
                model=model,
                dataset=dataset,
                task_name=task_name,
                total=total,
            )
        )
        correct = sum(record["correct"] for record in records)
        unparsed = sum(record["pred_answer"] is None for record in records)
        accuracy = correct / total if total else 0.0
        logger.info(
            "📊 %s accuracy=%.4f (%d/%d, unparsed=%d, %.1fs)",
            task_name,
            accuracy,
            correct,
            total,
            unparsed,
            eval_seconds,
        )
        return TaskEvaluation(
            task=task_name,
            accuracy=accuracy,
            correct=correct,
            total=total,
            unparsed=unparsed,
            eval_seconds=eval_seconds,
            evaluation_protocol=self._checkers[task_name].evaluation_protocol,
            records=records,
        )

    def _generate_and_check(
        self,
        *,
        model: PreTrainedModel,
        dataset: Dataset,
        task_name: str,
        total: int,
    ) -> list[dict[str, Any]]:
        checker = self._checkers[task_name]
        records = []
        old_padding_side = self._tokenizer.padding_side
        old_truncation_side = self._tokenizer.truncation_side
        self._tokenizer.padding_side = LEFT_SIDE
        self._tokenizer.truncation_side = LEFT_SIDE

        try:
            with create_progress(
                stage=Stage(
                    emoji=EVALUATE.emoji,
                    title=f"{EVALUATE.title} {task_name}",
                    colour=EVALUATE.colour,
                ),
                total=total,
                unit="ex",
            ) as bar:
                for start in range(0, total, self._batch_size):
                    rows = dataset[start : min(start + self._batch_size, total)]
                    prompts = [str(instruction) for instruction in rows["instruction"]]
                    completions = self._generate(
                        model=model,
                        prompts=prompts,
                    )
                    for prompt, completion, gold_raw in zip(prompts, completions, rows["gold_answer"]):
                        check = checker.check(
                            completion=completion,
                            gold_raw=gold_raw,
                        )
                        records.append(
                            {
                                "task": task_name,
                                "instruction": prompt,
                                "prediction_text": completion,
                                "pred_answer": check.pred_answer,
                                "gold_answer": check.gold_answer,
                                "correct": bool(check.correct),
                                "extraction_source": check.extraction_source,
                            }
                        )
                    bar.update(len(prompts))
                    bar.set_postfix_str(f"✅ {sum(record['correct'] for record in records)}/{len(records)}")
        finally:
            self._tokenizer.padding_side = old_padding_side
            self._tokenizer.truncation_side = old_truncation_side
        return records

    def _generate(
        self,
        *,
        model: PreTrainedModel,
        prompts: list[str],
    ) -> list[str]:
        encoded = self._tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=self._max_new_tokens,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        prompt_width = encoded["input_ids"].shape[1]
        return self._tokenizer.batch_decode(
            generated[:, prompt_width:],
            skip_special_tokens=True,
        )
