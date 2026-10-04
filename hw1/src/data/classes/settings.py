from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.2-1B"
TRAIN_DATASET_ID = "meta-math/MetaMathQA"
EVAL_DATASET_ID = "fxmeng/pissa-dataset"
EVAL_DATA_DIR = "metamath"
EVAL_TASKS = ("gsm8k", "math")
DEFAULT_COMPONENT = "default"
QUARTER_COMPONENT = "p25"
HALF_COMPONENT = "p50"
BOTTOM_COMPONENT = "bottom"
COMPONENT_LABELS = {
    DEFAULT_COMPONENT: "Top-16",
    QUARTER_COMPONENT: "25% position",
    HALF_COMPONENT: "50% position",
    BOTTOM_COMPONENT: "Bottom-16",
}


@dataclass(frozen=True, slots=True)
class RunSpec:
    rank: int
    component: str = DEFAULT_COMPONENT


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    result_root: Path
    model_id: str = MODEL_ID
    train_dataset_id: str = TRAIN_DATASET_ID
    train_limit: int | None = 25000
    eval_dataset_id: str = EVAL_DATASET_ID
    eval_data_dir: str = EVAL_DATA_DIR
    eval_tasks: tuple[str, ...] = EVAL_TASKS
    max_length: int = 512
    epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    target_global_batch: int = 128
    micro_batch_size: int = 8
    eval_max_new_tokens: int = 256
    eval_batch_size: int = 4
    max_eval_examples: int | None = 1000
    seed: int = 42
    train_device: str = "cuda:0"
    svd_device: str = "cuda:0"
    use_data_parallel: bool = False
    data_parallel_device_ids: tuple[int, ...] = (0, 1)
    save_predictions: bool = True
    save_adapters: bool = True
    save_full_model: bool = True
    oom_retries: int = 2

    def __post_init__(self) -> None:
        if self.target_global_batch % self.micro_batch_size != 0:
            raise ValueError(
                f"target_global_batch={self.target_global_batch} must be divisible by "
                f"micro_batch_size={self.micro_batch_size}"
            )

    @property
    def grad_accum_steps(self) -> int:
        return max(1, self.target_global_batch // self.micro_batch_size)

    @property
    def effective_batch(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps
