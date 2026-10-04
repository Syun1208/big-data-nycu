from __future__ import annotations

import torch
import torch.nn as nn


class LossOnlyWrapper(nn.Module):
    def __init__(self, *, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        return self.base_model(**batch).loss
