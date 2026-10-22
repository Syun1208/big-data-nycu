from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.data.classes.samples import CutoffSample
from src.data.classes.settings import EVENTS, STHNSettings
from src.models.sequence_model import BatchedSequenceModel, SequenceTensors
from src.services.interaction_sequences import InteractionSequenceBuilder
from src.services.scoring import ThresholdTuner

WEIGHTS_FILE = "sthn.pt"
META_FILE = "sthn_meta.json"


class TimeEncode(nn.Module):
    def __init__(self, *, dims: int) -> None:
        super().__init__()
        frequencies = torch.from_numpy(1 / 10 ** np.linspace(0, 9, dims, dtype="float32"))
        self.register_buffer("frequencies", frequencies)

    def forward(self, age: torch.Tensor) -> torch.Tensor:
        return torch.cos(age.unsqueeze(-1) * self.frequencies)


class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dims: int,
        expansion_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.linear_0 = nn.Linear(dims, expansion_factor * dims)
        self.linear_1 = nn.Linear(expansion_factor * dims, dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear_1(self.dropout(F.gelu(self.linear_0(x)))))


class PatchAttention(nn.Module):
    def __init__(
        self,
        *,
        dims: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if dims % num_heads:
            raise ValueError(f"hidden_dims {dims} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.dropout = dropout
        self.query = nn.Linear(dims, dims)
        self.key = nn.Linear(dims, dims)
        self.value = nn.Linear(dims, dims)
        self.out = nn.Sequential(nn.Linear(dims, dims), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dims = x.shape
        shape = (batch, tokens, self.num_heads, dims // self.num_heads)
        query, key, value = (layer(x).view(shape).transpose(1, 2) for layer in (self.query, self.key, self.value))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(attended.transpose(1, 2).reshape(batch, tokens, dims))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dims: int,
        num_heads: int,
        expansion_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_mixer = PatchAttention(
            dims=dims,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.channel_norm = nn.LayerNorm(dims)
        self.channel_mixer = FeedForward(
            dims=dims,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(x)
        return x + self.channel_mixer(self.channel_norm(x))


def positional_encoding(
    *,
    length: int,
    dims: int,
) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    inverse = torch.exp(torch.arange(0, dims, 2, dtype=torch.float32) * (-math.log(10000.0) / dims))
    encoding = torch.zeros(length, dims)
    encoding[:, 0::2] = torch.sin(position * inverse)
    encoding[:, 1::2] = torch.cos(position * inverse)[:, : dims // 2]
    return encoding


class PatchEncoder(nn.Module):
    def __init__(
        self,
        *,
        max_edges: int,
        window_size: int,
        num_kinds: int,
        time_dims: int,
        hidden_dims: int,
        num_layers: int,
        num_heads: int,
        expansion_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if max_edges % window_size:
            raise ValueError(f"max_edges {max_edges} must be divisible by window_size {window_size}")
        self.num_kinds = num_kinds
        self.window_size = window_size
        self.time_encoder = TimeEncode(dims=time_dims)
        self.feat_encoder = nn.Linear(num_kinds + time_dims, hidden_dims)
        self.pad_projector = nn.Linear(window_size * hidden_dims, hidden_dims)
        self.register_buffer(
            "position",
            positional_encoding(
                length=max_edges // window_size,
                dims=hidden_dims,
            ),
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dims=hidden_dims,
                    num_heads=num_heads,
                    expansion_factor=expansion_factor,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(hidden_dims)
        self.mlp_head = nn.Linear(hidden_dims, hidden_dims)

    def forward(
        self,
        *,
        kinds: torch.Tensor,
        age_seconds: torch.Tensor,
    ) -> torch.Tensor:
        valid = kinds >= 0
        kind_feature = F.one_hot(kinds.clamp(min=0).long(), self.num_kinds).float()
        edges = self.feat_encoder(torch.cat([kind_feature, self.time_encoder(age_seconds)], dim=-1))
        edges = edges * valid.unsqueeze(-1)
        batch, max_edges, hidden = edges.shape
        x = self.pad_projector(edges.reshape(batch, max_edges // self.window_size, self.window_size * hidden))
        x = x + self.position
        for block in self.blocks:
            x = block(x)
        return self.mlp_head(self.layernorm(x).mean(dim=1))


class STHN(nn.Module):
    def __init__(
        self,
        *,
        max_edges: int,
        window_size: int,
        time_dims: int,
        hidden_dims: int,
        predictor_dims: int,
        num_layers: int,
        num_heads: int,
        expansion_factor: int,
        dropout: float,
        node_feature_dims: int,
    ) -> None:
        super().__init__()
        self.encoder = PatchEncoder(
            max_edges=max_edges,
            window_size=window_size,
            num_kinds=len(EVENTS),
            time_dims=time_dims,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )
        self.node_fc = nn.Linear(hidden_dims + node_feature_dims, predictor_dims)
        self.out_fc = nn.Linear(predictor_dims, len(EVENTS))

    def embed(
        self,
        *,
        kinds: torch.Tensor,
        age_seconds: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(
            kinds=kinds,
            age_seconds=age_seconds,
        )

    def forward(
        self,
        *,
        kinds: torch.Tensor,
        age_seconds: torch.Tensor,
        node_features: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.embed(
            kinds=kinds,
            age_seconds=age_seconds,
        )
        if node_features is not None:
            x = torch.cat([x, node_features], dim=1)
        return self.out_fc(F.relu(self.node_fc(x)))


class ComprehensiveSTHN(BatchedSequenceModel):
    base_name = "sthn"
    weights_file = WEIGHTS_FILE
    meta_file = META_FILE

    def __init__(
        self,
        *,
        settings: STHNSettings,
        sequence_builder: InteractionSequenceBuilder,
        tuner: ThresholdTuner,
    ) -> None:
        super().__init__(
            settings=settings,
            tuner=tuner,
        )
        self._sthn_settings = settings
        self._sequence_builder = sequence_builder

    def _new_network(self) -> STHN:
        settings = self._sthn_settings
        return STHN(
            max_edges=settings.max_edges,
            window_size=settings.window_size,
            time_dims=settings.time_dims,
            hidden_dims=settings.hidden_dims,
            predictor_dims=settings.predictor_dims,
            num_layers=settings.num_layers,
            num_heads=settings.num_heads,
            expansion_factor=settings.channel_expansion_factor,
            dropout=settings.dropout,
            node_feature_dims=len(self._columns),
        )

    def _sequence_tensors(self, *, samples: Sequence[CutoffSample]) -> SequenceTensors:
        sequences = [
            self._sequence_builder.build(
                user_ids=sample.dataset.meta["user_id"].to_numpy(dtype="int64"),
                feature_time=sample.cutoff.feature_time,
            )
            for sample in samples
        ]
        kinds = torch.from_numpy(np.concatenate([sequence.kinds for sequence in sequences])).to(self._device)
        ages = torch.from_numpy(np.concatenate([sequence.age_seconds for sequence in sequences])).to(self._device)
        return kinds, ages

    def _logits(self, *, sequences: SequenceTensors, features: torch.Tensor | None) -> torch.Tensor:
        kinds, ages = sequences
        return self._network(kinds=kinds, age_seconds=ages, node_features=features)

    def _embedding(self, *, sequences: SequenceTensors) -> torch.Tensor:
        kinds, ages = sequences
        return self._network.embed(kinds=kinds, age_seconds=ages)
