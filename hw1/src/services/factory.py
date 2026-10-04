from __future__ import annotations

import logging

import torch

from src.data.classes.settings import (
    BOTTOM_COMPONENT,
    DEFAULT_COMPONENT,
    HALF_COMPONENT,
    QUARTER_COMPONENT,
    ExperimentSettings,
)
from src.data.loaders.math_datasets import MetaMathTrainLoader, PiSSAEvalLoader
from src.data.loaders.supervised import ResponseOnlyCollator, SupervisedExampleTokenizer
from src.interface.answer_checker import AnswerChecker
from src.interface.singular_range import SingularRangeSelector
from src.models.loading import PretrainedModelLoader
from src.models.pissa_adapter import PiSSADecomposer, PiSSAInjector
from src.models.singular_ranges import BottomSingularRange, PositionSingularRange, TopSingularRange
from src.services.answer_checkers import Gsm8kAnswerChecker, MathAnswerChecker
from src.services.evaluator import GenerationEvaluator
from src.services.experiment_runner import ExperimentRunner
from src.services.result_store import ResultStore
from src.services.trainer import PiSSATrainer, TrainingSettings
from src.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def build_range_selectors() -> dict[str, SingularRangeSelector]:
    return {
        DEFAULT_COMPONENT: TopSingularRange(),
        QUARTER_COMPONENT: PositionSingularRange(
            numerator=1,
            denominator=4,
        ),
        HALF_COMPONENT: PositionSingularRange(
            numerator=1,
            denominator=2,
        ),
        BOTTOM_COMPONENT: BottomSingularRange(),
    }


def build_answer_checkers() -> dict[str, AnswerChecker]:
    return {
        "gsm8k": Gsm8kAnswerChecker(),
        "math": MathAnswerChecker(),
    }


def build_training_settings(*, settings: ExperimentSettings) -> TrainingSettings:
    return TrainingSettings(
        epochs=settings.epochs,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        warmup_ratio=settings.warmup_ratio,
        micro_batch_size=settings.micro_batch_size,
        grad_accum_steps=settings.grad_accum_steps,
        seed=settings.seed,
        use_data_parallel=settings.use_data_parallel,
        data_parallel_device_ids=settings.data_parallel_device_ids,
    )


def build_experiment_runner(*, settings: ExperimentSettings) -> ExperimentRunner:
    seed_everything(seed=settings.seed)
    train_device = torch.device(settings.train_device)

    model_loader = PretrainedModelLoader(
        model_id=settings.model_id,
        device=train_device,
    )
    tokenizer = model_loader.load_tokenizer(max_length=settings.max_length)

    train_dataset = MetaMathTrainLoader(
        dataset_id=settings.train_dataset_id,
        limit=settings.train_limit,
    ).load()
    eval_datasets = PiSSAEvalLoader(
        dataset_id=settings.eval_dataset_id,
        data_dir=settings.eval_data_dir,
        tasks=settings.eval_tasks,
    ).load()
    logger.info("📥 Train rows: %d", len(train_dataset))
    for task, dataset in eval_datasets.items():
        logger.info("📥 Eval %s rows: %d", task, len(dataset))

    tokenized_train = SupervisedExampleTokenizer(
        tokenizer=tokenizer,
        max_length=settings.max_length,
    ).tokenize_dataset(dataset=train_dataset)

    checkers = build_answer_checkers()
    return ExperimentRunner(
        settings=settings,
        model_loader=model_loader,
        tokenizer=tokenizer,
        injector=PiSSAInjector(
            decomposer=PiSSADecomposer(
                svd_device=torch.device(settings.svd_device),
                range_selectors=build_range_selectors(),
            )
        ),
        trainer=PiSSATrainer(
            settings=build_training_settings(settings=settings),
            collator=ResponseOnlyCollator(pad_token_id=tokenizer.pad_token_id),
            device=train_device,
        ),
        evaluator=GenerationEvaluator(
            tokenizer=tokenizer,
            checkers={task: checkers[task] for task in settings.eval_tasks},
            device=train_device,
            max_length=settings.max_length,
            max_new_tokens=settings.eval_max_new_tokens,
            batch_size=settings.eval_batch_size,
        ),
        store=ResultStore(
            root=settings.result_root,
            save_predictions=settings.save_predictions,
        ),
        train_dataset=tokenized_train,
        eval_datasets=eval_datasets,
    )
