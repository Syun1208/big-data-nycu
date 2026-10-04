from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PiSSALinear(nn.Module):
    def __init__(
        self,
        *,
        weight_residual: torch.Tensor,
        pissa_A: torch.Tensor,
        pissa_B: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        out_features, in_features = weight_residual.shape
        rank = pissa_A.shape[1]
        if pissa_A.shape != (out_features, rank) or pissa_B.shape != (rank, in_features):
            raise ValueError(
                f"Incompatible PiSSA shapes: W={tuple(weight_residual.shape)}, "
                f"A={tuple(pissa_A.shape)}, B={tuple(pissa_B.shape)}"
            )

        self.rank = rank
        self.weight = nn.Parameter(
            weight_residual.contiguous(),
            requires_grad=False,
        )
        self.pissa_A = nn.Parameter(pissa_A.float())
        self.pissa_B = nn.Parameter(pissa_B.float())
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(
                bias.detach().clone(),
                requires_grad=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen_output = F.linear(x, self.weight, self.bias)
        with torch.autocast(
            device_type=x.device.type,
            enabled=False,
        ):
            hidden = F.linear(x.float(), self.pissa_B)
            update = F.linear(hidden, self.pissa_A)
        return frozen_output + update.to(frozen_output.dtype)
