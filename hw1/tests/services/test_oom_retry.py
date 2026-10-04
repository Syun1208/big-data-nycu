from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from src.data.classes.settings import ExperimentSettings, RunSpec
from src.services.experiment_runner import ExperimentRunner
from src.services.trainer import PiSSATrainer, TrainingSettings

GLOBAL_BATCH = 128


def build_trainer(*, micro_batch_size: int) -> PiSSATrainer:
    return PiSSATrainer(
        settings=TrainingSettings(
            epochs=1,
            learning_rate=2e-5,
            weight_decay=0.0,
            warmup_ratio=0.03,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=GLOBAL_BATCH // micro_batch_size,
            seed=42,
            use_data_parallel=False,
            data_parallel_device_ids=(0, 1),
        ),
        collator=lambda features: {},
        device=torch.device("cpu"),
    )


def build_runner(
    *,
    tmp_path: Path,
    oom_retries: int,
) -> ExperimentRunner:
    return ExperimentRunner(
        settings=ExperimentSettings(
            result_root=tmp_path,
            train_device="cpu",
            oom_retries=oom_retries,
        ),
        model_loader=None,
        tokenizer=None,
        injector=None,
        trainer=build_trainer(micro_batch_size=8),
        evaluator=None,
        store=None,
        train_dataset=None,
        eval_datasets={},
    )


def test_with_micro_batch_size_keeps_global_batch() -> None:
    smaller = build_trainer(micro_batch_size=8).with_micro_batch_size(micro_batch_size=4)
    assert (smaller.micro_batch_size, smaller.grad_accum_steps) == (4, 32)
    assert smaller.micro_batch_size * smaller.grad_accum_steps == GLOBAL_BATCH


def test_oom_retry_halves_micro_batch_until_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = build_runner(
        tmp_path=tmp_path,
        oom_retries=2,
    )
    attempts: list[int] = []

    def fake_run(*, spec: RunSpec, trainer: PiSSATrainer) -> dict[str, Any]:
        attempts.append(trainer.micro_batch_size)
        if trainer.micro_batch_size > 2:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        return {"status": "ok", "micro_batch_size": trainer.micro_batch_size}

    monkeypatch.setattr(runner, "run_pissa_experiment", fake_run)
    row = runner._run_with_oom_retry(spec=RunSpec(rank=256))
    assert attempts == [8, 4, 2]
    assert row == {"status": "ok", "micro_batch_size": 2}


def test_oom_recorded_after_retries_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = build_runner(
        tmp_path=tmp_path,
        oom_retries=1,
    )

    def always_oom(*, spec: RunSpec, trainer: PiSSATrainer) -> dict[str, Any]:
        raise torch.cuda.OutOfMemoryError("fake OOM")

    monkeypatch.setattr(runner, "run_pissa_experiment", always_oom)
    row = runner._run_with_oom_retry(spec=RunSpec(rank=256))
    assert row["status"] == "oom"
    assert row["run_tag"] == "metamath_pissa_default_r256"
