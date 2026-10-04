from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from src.utils.cuda_memory import release_cuda_memory

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

MODEL_DTYPE = torch.float16
CPU_DEVICE = "cpu"


class PretrainedModelLoader:
    def __init__(
        self,
        *,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype = MODEL_DTYPE,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._dtype = dtype

    @property
    def model_id(self) -> str:
        return self._model_id

    def load_tokenizer(self, *, max_length: int) -> PreTrainedTokenizerBase:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = max_length
        logger.info(
            "📝 Tokenizer %s (EOS=%s, PAD=%s)",
            self._model_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )
        return tokenizer

    def load_model(self, *, to_device: bool) -> PreTrainedModel:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            dtype=self._dtype,
            device_map=CPU_DEVICE,
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = True
        if to_device:
            model = model.to(self._device)
        return model


def release_model(*, model: torch.nn.Module | None) -> None:
    if model is not None:
        try:
            model.to(CPU_DEVICE)
        except Exception:
            logger.debug("Could not move model to CPU before release", exc_info=True)
    release_cuda_memory()
