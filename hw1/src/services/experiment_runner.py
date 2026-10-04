from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import torch

from src.data.classes.records import FLATTENED_EVALUATION_KEYS, TaskEvaluation
from src.data.classes.settings import ExperimentSettings, RunSpec
from src.models.loading import PretrainedModelLoader, release_model
from src.models.pissa_adapter import PiSSAInjector, has_pissa_layers, verify_trainables
from src.services.evaluator import GenerationEvaluator
from src.services.result_store import FRESH_METHOD, PISSA_METHOD, ResultStore, build_run_tag
from src.services.trainer import PiSSATrainer
from src.utils.cuda_memory import release_cuda_memory, time_cuda_call

if TYPE_CHECKING:
    import pandas as pd
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

TRAIN_TASK_NAME = "metamath"
SECONDS_PER_MINUTE = 60.0
SEPARATOR = "=" * 80


def flatten_evaluations(
    *,
    evaluations: Mapping[str, TaskEvaluation],
    prefix: str = "",
) -> dict[str, Any]:
    flat = {}
    for task, evaluation in evaluations.items():
        values = asdict(evaluation)
        for key in FLATTENED_EVALUATION_KEYS:
            flat[f"{prefix}{task}_{key}"] = values[key]
    return flat


def combine_records(*, evaluations: Mapping[str, TaskEvaluation]) -> list[dict[str, Any]]:
    return [record for evaluation in evaluations.values() for record in evaluation.records]


class ExperimentRunner:
    def __init__(
        self,
        *,
        settings: ExperimentSettings,
        model_loader: PretrainedModelLoader,
        tokenizer: PreTrainedTokenizerBase,
        injector: PiSSAInjector,
        trainer: PiSSATrainer,
        evaluator: GenerationEvaluator,
        store: ResultStore,
        train_dataset: Dataset,
        eval_datasets: Mapping[str, Dataset],
    ) -> None:
        self._settings = settings
        self._model_loader = model_loader
        self._tokenizer = tokenizer
        self._injector = injector
        self._trainer = trainer
        self._evaluator = evaluator
        self._store = store
        self._train_dataset = train_dataset
        self._eval_datasets = dict(eval_datasets)
        self._device = torch.device(settings.train_device)

    def run_all(
        self,
        *,
        run_specs: Sequence[RunSpec],
        run_fresh_baseline: bool,
    ) -> pd.DataFrame:
        results: list[dict[str, Any]] = []

        if run_fresh_baseline:
            try:
                results.append(self.run_fresh_baseline())
            except Exception as error:
                logger.exception("Fresh baseline failed: %r", error)
                results.append(
                    self._error_row(
                        spec=None,
                        status="error",
                        error=error,
                    )
                )
            self._store.save_master_results(rows=results)

        for spec in run_specs:
            try:
                results.append(self._run_with_oom_retry(spec=spec))
            finally:
                self._store.save_master_results(rows=results)

        frame = self._store.save_master_results(rows=results)
        logger.info("💾 Benchmark results: %s", self._store.benchmark_csv_path)
        return frame

    def _run_with_oom_retry(self, *, spec: RunSpec) -> dict[str, Any]:
        trainer = self._trainer
        for attempt in range(self._settings.oom_retries + 1):
            try:
                return self.run_pissa_experiment(
                    spec=spec,
                    trainer=trainer,
                )
            except torch.cuda.OutOfMemoryError as error:
                release_cuda_memory()
                can_retry = attempt < self._settings.oom_retries and trainer.micro_batch_size > 1
                if not can_retry:
                    logger.exception("OOM recorded; continuing: %s: %r", spec, error)
                    return self._error_row(
                        spec=spec,
                        status="oom",
                        error=error,
                    )
                trainer = trainer.with_micro_batch_size(micro_batch_size=trainer.micro_batch_size // 2)
                logger.warning(
                    "OOM on %s; retry %d/%d with micro_batch_size=%d, grad_accum_steps=%d (global batch unchanged)",
                    spec,
                    attempt + 1,
                    self._settings.oom_retries,
                    trainer.micro_batch_size,
                    trainer.grad_accum_steps,
                )
                logger.debug("OOM detail", exc_info=True)
            except Exception as error:
                logger.exception("Error recorded; continuing: %s: %r", spec, error)
                return self._error_row(
                    spec=spec,
                    status="error",
                    error=error,
                )
        raise AssertionError("unreachable")

    def run_fresh_baseline(self) -> dict[str, Any]:
        tag = build_run_tag(method=FRESH_METHOD)
        self._log_banner(title="FRESH BASELINE: evaluate GSM8K + MATH")

        model = self._model_loader.load_model(to_device=True)
        try:
            evaluations = self._evaluate(model=model)
            summary = {
                "run_tag": tag,
                "training_dataset": self._settings.train_dataset_id,
                "training_examples": 0,
                "method": FRESH_METHOD,
                "rank": None,
                "component": None,
                **flatten_evaluations(evaluations=evaluations),
                "init_seconds": 0.0,
                "trainable_parameters": 0,
                "status": "ok",
                "seed": self._settings.seed,
            }
            self._store.save_summary(
                summary=summary,
                records=combine_records(evaluations=evaluations),
            )
            return summary
        finally:
            release_model(model=model)

    def run_pissa_experiment(
        self,
        *,
        spec: RunSpec,
        trainer: PiSSATrainer | None = None,
    ) -> dict[str, Any]:
        trainer = trainer or self._trainer
        tag = build_run_tag(
            method=PISSA_METHOD,
            rank=spec.rank,
            component=spec.component,
        )
        self._log_banner(title=tag)

        model = self._model_loader.load_model(to_device=False)
        try:
            torch.manual_seed(self._settings.seed)
            torch.cuda.manual_seed_all(self._settings.seed)

            init_stats, init_seconds = time_cuda_call(
                action=lambda: self._injector.inject(
                    model=model,
                    rank=spec.rank,
                    component=spec.component,
                )
            )
            trainable, total_parameters = verify_trainables(model=model)
            model = model.to(self._device)

            outcome = trainer.train(
                model=model,
                train_dataset=self._train_dataset,
                task_name=TRAIN_TASK_NAME,
            )
            history = outcome.history
            history_path = self._save_history(
                tag=tag,
                spec=spec,
                history=history,
            )

            partial = {
                "run_tag": tag,
                "training_dataset": self._settings.train_dataset_id,
                "training_examples": len(self._train_dataset),
                "method": PISSA_METHOD,
                "rank": spec.rank,
                "component": spec.component,
                "init_seconds": init_seconds,
                "micro_batch_size": trainer.micro_batch_size,
                "grad_accum_steps": trainer.grad_accum_steps,
                "train_seconds": outcome.train_seconds,
                "train_minutes": outcome.train_seconds / SECONDS_PER_MINUTE,
                "trainable_parameters": trainable,
                "status": "trained_pending_merge_eval",
                "seed": self._settings.seed,
            }
            self._store.save_training_time_partial(summary=partial)
            if self._settings.save_adapters:
                self._save_adapter(
                    tag=tag,
                    model=model,
                    metadata=partial,
                )

            merged_names, merge_seconds = time_cuda_call(action=lambda: self._injector.merge_and_unload(model=model))
            if has_pissa_layers(model=model):
                raise RuntimeError("Evaluation must use the merged standalone model.")
            logger.info("🔗 PiSSA merge time: %.2fs", merge_seconds)

            evaluations = self._evaluate(model=model)

            full_model_path = self._store.full_model_path(tag=tag)
            if self._settings.save_full_model:
                model = model.to("cpu")
                release_cuda_memory()
                model.save_pretrained(
                    full_model_path,
                    safe_serialization=True,
                )
                self._tokenizer.save_pretrained(full_model_path)
                logger.info("💾 Merged full model saved: %s", full_model_path)

            summary = {
                **partial,
                **flatten_evaluations(evaluations=evaluations),
                "merge_seconds": merge_seconds,
                "merged_layer_count": len(merged_names),
                "evaluated_after_merge": True,
                "full_model_saved": bool(self._settings.save_full_model),
                "full_model_path": str(full_model_path) if self._settings.save_full_model else None,
                "status": "ok",
                "final_training_loss": history[-1]["loss"] if history else None,
                "training_history_path": str(history_path),
                "total_parameters_with_adapter": total_parameters,
                "max_pissa_reconstruction_error": max(stats.relative_error_stored for stats in init_stats),
            }
            self._store.save_summary(
                summary=summary,
                records=combine_records(evaluations=evaluations),
            )
            return summary
        finally:
            release_model(model=model)

    def _evaluate(self, *, model: torch.nn.Module) -> dict[str, TaskEvaluation]:
        return self._evaluator.evaluate_all(
            model=model,
            datasets=self._eval_datasets,
            max_examples=self._settings.max_eval_examples,
        )

    def _save_history(
        self,
        *,
        tag: str,
        spec: RunSpec,
        history: Sequence[Mapping[str, Any]],
    ) -> str:
        import pandas as pd

        frame = pd.DataFrame(list(history))
        frame["training_dataset"] = self._settings.train_dataset_id
        frame["method"] = PISSA_METHOD
        frame["rank"] = spec.rank
        frame["component"] = spec.component
        return str(
            self._store.save_training_history(
                tag=tag,
                history=frame,
            )
        )

    def _save_adapter(
        self,
        *,
        tag: str,
        model: torch.nn.Module,
        metadata: Mapping[str, Any],
    ) -> None:
        adapter_state = {
            name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        path = self._store.adapter_path(tag=tag)
        torch.save(
            {"metadata": dict(metadata), "state_dict": adapter_state},
            path,
        )
        logger.info("💾 PiSSA adapter saved: %s", path)

    def _error_row(
        self,
        *,
        spec: RunSpec | None,
        status: str,
        error: Exception,
    ) -> dict[str, Any]:
        if spec is None:
            return {
                "run_tag": build_run_tag(method=FRESH_METHOD),
                "training_dataset": self._settings.train_dataset_id,
                "method": FRESH_METHOD,
                "rank": None,
                "component": None,
                "status": status,
                "error": repr(error),
            }
        return {
            "rank": spec.rank,
            "component": spec.component,
            "training_dataset": self._settings.train_dataset_id,
            "method": PISSA_METHOD,
            "run_tag": build_run_tag(
                method=PISSA_METHOD,
                rank=spec.rank,
                component=spec.component,
            ),
            "status": status,
            "error": repr(error),
        }

    @staticmethod
    def _log_banner(*, title: str) -> None:
        logger.info(SEPARATOR)
        logger.info("🧭 %s", title)
        logger.info(SEPARATOR)
