from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.loss_wrapper import LossOnlyWrapper
from src.utils.cuda_memory import time_cuda_call
from src.utils.progress import Stage, create_progress
from src.utils.seeding import seed_everything

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

TRAIN = Stage(emoji="🚀", title="Train", colour="green")
SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    micro_batch_size: int
    grad_accum_steps: int
    seed: int
    use_data_parallel: bool
    data_parallel_device_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    history: list[dict[str, Any]]
    train_seconds: float


@dataclass(frozen=True, slots=True)
class OptimizationPlan:
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    total_updates: int
    warmup_updates: int


class WarmupCosineSchedule:
    def __init__(
        self,
        *,
        warmup_updates: int,
        total_updates: int,
    ) -> None:
        self._warmup_updates = warmup_updates
        self._total_updates = total_updates

    def __call__(self, step: int) -> float:
        if step < self._warmup_updates:
            return float(step + 1) / float(max(1, self._warmup_updates))
        progress = (step - self._warmup_updates) / max(1, self._total_updates - self._warmup_updates)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class PiSSATrainer:
    def __init__(
        self,
        *,
        settings: TrainingSettings,
        collator: Callable[[Sequence[Mapping[str, list[int]]]], dict[str, torch.Tensor]],
        device: torch.device,
    ) -> None:
        self._settings = settings
        self._collator = collator
        self._device = device

    @property
    def micro_batch_size(self) -> int:
        return self._settings.micro_batch_size

    @property
    def grad_accum_steps(self) -> int:
        return self._settings.grad_accum_steps

    def with_micro_batch_size(self, *, micro_batch_size: int) -> PiSSATrainer:
        global_batch = self._settings.micro_batch_size * self._settings.grad_accum_steps
        if global_batch % micro_batch_size != 0:
            raise ValueError(f"micro_batch_size={micro_batch_size} does not divide global batch {global_batch}")
        return PiSSATrainer(
            settings=replace(
                self._settings,
                micro_batch_size=micro_batch_size,
                grad_accum_steps=global_batch // micro_batch_size,
            ),
            collator=self._collator,
            device=self._device,
        )

    def train(
        self,
        *,
        model: PreTrainedModel,
        train_dataset: Dataset,
        task_name: str,
    ) -> TrainingOutcome:
        seed_everything(seed=self._settings.seed)
        loader = DataLoader(
            dataset=train_dataset,
            batch_size=self._settings.micro_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self._settings.seed),
            collate_fn=self._collator,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        plan = self._build_optimization_plan(
            model=model,
            num_batches=len(loader),
        )
        logger.info(
            "🚀 Training %s: %d examples, %d micro-batches, %d updates, %d warmup updates.",
            task_name,
            len(train_dataset),
            len(loader),
            plan.total_updates,
            plan.warmup_updates,
        )

        model.config.use_cache = False
        model.train()
        train_model = self._wrap_for_training(model=model)

        history, train_seconds = time_cuda_call(
            action=lambda: self._run_epochs(
                train_model=train_model,
                loader=loader,
                plan=plan,
                task_name=task_name,
            )
        )
        logger.info("🚀 Training time: %.2f min", train_seconds / SECONDS_PER_MINUTE)

        model.config.use_cache = True
        return TrainingOutcome(
            history=history,
            train_seconds=train_seconds,
        )

    def _run_epochs(
        self,
        *,
        train_model: nn.Module,
        loader: DataLoader,
        plan: OptimizationPlan,
        task_name: str,
    ) -> list[dict[str, Any]]:
        grad_accum_steps = self._settings.grad_accum_steps
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
        plan.optimizer.zero_grad(set_to_none=True)
        history = []
        update_step = 0
        running_loss = 0.0

        for epoch in range(self._settings.epochs):
            with create_progress(
                stage=Stage(
                    emoji=TRAIN.emoji,
                    title=f"{TRAIN.title} {task_name} epoch {epoch + 1}/{self._settings.epochs}",
                    colour=TRAIN.colour,
                ),
                total=len(loader),
                unit="batch",
                iterable=loader,
            ) as bar:
                for micro_step, batch in enumerate(bar, 1):
                    batch = {key: value.to(self._device, non_blocking=True) for key, value in batch.items()}

                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    ):
                        loss = train_model(**batch)
                        if loss.ndim > 0:
                            loss = loss.mean()
                        scaled_loss = loss / grad_accum_steps

                    scaler.scale(scaled_loss).backward()
                    running_loss += float(loss.detach().cpu())

                    is_accumulation_boundary = micro_step % grad_accum_steps == 0
                    if not (is_accumulation_boundary or micro_step == len(loader)):
                        continue

                    scaler.step(plan.optimizer)
                    scaler.update()
                    plan.optimizer.zero_grad(set_to_none=True)
                    plan.scheduler.step()
                    update_step += 1

                    accumulated_steps = (
                        grad_accum_steps
                        if is_accumulation_boundary
                        else micro_step - (update_step - 1) * grad_accum_steps
                    )
                    mean_loss = running_loss / accumulated_steps
                    history.append(
                        {
                            "update": update_step,
                            "loss": mean_loss,
                            "lr": plan.scheduler.get_last_lr()[0],
                        }
                    )
                    running_loss = 0.0
                    bar.set_postfix_str(f"loss={mean_loss:.4f} update={update_step}")
                    logger.debug("update=%d loss=%.4f lr=%.3e", update_step, mean_loss, history[-1]["lr"])
        return history

    def _wrap_for_training(self, *, model: PreTrainedModel) -> nn.Module:
        loss_model = LossOnlyWrapper(base_model=model)
        if not self._settings.use_data_parallel:
            return loss_model
        return nn.DataParallel(
            module=loss_model,
            device_ids=list(self._settings.data_parallel_device_ids),
            output_device=self._settings.data_parallel_device_ids[0],
        )

    def _build_optimization_plan(
        self,
        *,
        model: nn.Module,
        num_batches: int,
    ) -> OptimizationPlan:
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        updates_per_epoch = math.ceil(num_batches / self._settings.grad_accum_steps)
        total_updates = self._settings.epochs * updates_per_epoch
        warmup_updates = max(1, math.ceil(total_updates * self._settings.warmup_ratio))

        optimizer = torch.optim.AdamW(
            params=trainable,
            lr=self._settings.learning_rate,
            weight_decay=self._settings.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=WarmupCosineSchedule(
                warmup_updates=warmup_updates,
                total_updates=total_updates,
            ),
        )
        return OptimizationPlan(
            optimizer=optimizer,
            scheduler=scheduler,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
        )
