from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from src.data.classes.samples import CutoffSample
from src.data.classes.settings import STGNNSettings
from src.models.sequence_model import BatchedSequenceModel, SequenceTensors
from src.services.activity_series import ActivitySeriesBuilder
from src.services.scoring import ThresholdTuner

WEIGHTS_FILE = "stgnn.pt"
META_FILE = "stgnn_meta.json"


class TemporalTransform(nn.Module):
    def __init__(
        self,
        *,
        dims: int,
        head_dims: int,
    ) -> None:
        super().__init__()
        self.head_dims = head_dims
        self.query = nn.Linear(dims, dims)
        self.key = nn.Linear(dims, dims)
        self.value = nn.Linear(dims, dims)
        self.norm = nn.LayerNorm(dims)
        self.feed_forward = nn.Sequential(nn.Linear(dims, dims), nn.ReLU(), nn.Linear(dims, dims))
        self.feed_forward_norm = nn.LayerNorm(dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        query = torch.cat(torch.split(self.query(x), self.head_dims, -1), 0).permute(0, 2, 1, 3)
        key = torch.cat(torch.split(self.key(x), self.head_dims, -1), 0).permute(0, 2, 3, 1)
        value = torch.cat(torch.split(self.value(x), self.head_dims, -1), 0).permute(0, 2, 1, 3)
        attention = torch.softmax(torch.matmul(query, key) / self.head_dims**0.5, -1)
        attended = torch.cat(torch.split(torch.matmul(attention, value), batch, 0), -1).permute(0, 2, 1, 3)
        attended = self.norm(attended + x)
        return self.feed_forward_norm(self.feed_forward(attended) + attended)


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
    return encoding.unsqueeze(0).unsqueeze(2)


class SpatialGNN(nn.Module):
    def __init__(self, *, dims: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(dims, dims), nn.Linear(dims, dims))
        self.update = nn.Linear(dims, dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        relation = torch.relu(torch.softmax(torch.matmul(projected, projected.transpose(-1, -2)), -1))
        relation = relation + torch.eye(x.shape[1], device=x.device)
        inverse_sqrt = relation.sum(-1).pow(-0.5)
        inverse_sqrt = torch.where(torch.isinf(inverse_sqrt), torch.zeros_like(inverse_sqrt), inverse_sqrt)
        adjacency = inverse_sqrt.unsqueeze(-1) * relation * inverse_sqrt.unsqueeze(-2)
        return torch.relu(self.update(torch.matmul(adjacency, x)))


class GraphGRUCell(nn.Module):
    def __init__(self, *, dims: int) -> None:
        super().__init__()
        self.dims = dims
        self.gates = nn.Linear(2 * dims, 2 * dims)
        self.candidate = nn.Linear(2 * dims, dims)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        reset, update = torch.split(torch.sigmoid(self.gates(torch.cat([x, hidden], -1))), self.dims, -1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * hidden], -1)))
        return update * candidate + (1 - update) * hidden


class SpatialTemporalGRU(nn.Module):
    def __init__(
        self,
        *,
        dims: int,
        num_steps: int,
    ) -> None:
        super().__init__()
        self.input_graphs = nn.ModuleList([SpatialGNN(dims=dims) for _ in range(num_steps)])
        self.hidden_graphs = nn.ModuleList([SpatialGNN(dims=dims) for _ in range(num_steps)])
        self.cells = nn.ModuleList([GraphGRUCell(dims=dims) for _ in range(num_steps)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, dims = x.shape
        hidden = torch.zeros(batch, nodes, dims, device=x.device)
        outputs = []
        for step in range(steps):
            graph_input = self.input_graphs[step](x[:, step])
            graph_hidden = self.hidden_graphs[step](hidden) if step else hidden
            hidden = self.cells[step](graph_input, graph_hidden)
            outputs.append(hidden)
        return torch.stack(outputs, 1)


class STGNN(nn.Module):
    def __init__(
        self,
        *,
        num_steps: int,
        hidden_dims: int,
        num_heads: int,
        num_layers: int,
        node_feature_dims: int,
    ) -> None:
        super().__init__()
        if hidden_dims % num_heads:
            raise ValueError(f"hidden_dims {hidden_dims} must be divisible by num_heads {num_heads}")
        self.start_embedding = nn.Linear(1, hidden_dims)
        self.recurrent = nn.ModuleList([SpatialTemporalGRU(dims=hidden_dims, num_steps=num_steps) for _ in range(num_layers)])
        self.register_buffer("position", positional_encoding(length=num_steps, dims=hidden_dims))
        self.transforms = nn.ModuleList([TemporalTransform(dims=hidden_dims, head_dims=hidden_dims // num_heads) for _ in range(num_layers)])
        self.user_projection = nn.Linear(node_feature_dims, hidden_dims) if node_feature_dims else None
        self.output = nn.Linear(hidden_dims * (2 if node_feature_dims else 1), 1)

    def node_states(self, *, series: torch.Tensor) -> torch.Tensor:
        x = self.start_embedding(series.unsqueeze(-1))
        for layer in self.recurrent:
            x = layer(x)
        x = x + self.position
        for transform in self.transforms:
            x = transform(x)
        return x[:, -1]

    def embed(self, *, series: torch.Tensor) -> torch.Tensor:
        return self.node_states(series=series).flatten(1)

    def forward(
        self,
        *,
        series: torch.Tensor,
        node_features: torch.Tensor | None,
    ) -> torch.Tensor:
        states = self.node_states(series=series)
        if self.user_projection is not None and node_features is not None:
            user = self.user_projection(node_features).unsqueeze(1).expand(-1, states.shape[1], -1)
            states = torch.cat([states, user], dim=-1)
        return self.output(states).squeeze(-1)


class ComprehensiveSTGNN(BatchedSequenceModel):
    base_name = "stgnn"
    weights_file = WEIGHTS_FILE
    meta_file = META_FILE

    def __init__(
        self,
        *,
        settings: STGNNSettings,
        series_builder: ActivitySeriesBuilder,
        tuner: ThresholdTuner,
    ) -> None:
        super().__init__(
            settings=settings,
            tuner=tuner,
        )
        self._stgnn_settings = settings
        self._series_builder = series_builder

    def _new_network(self) -> STGNN:
        settings = self._stgnn_settings
        return STGNN(
            num_steps=settings.num_steps,
            hidden_dims=settings.hidden_dims,
            num_heads=settings.num_heads,
            num_layers=settings.num_layers,
            node_feature_dims=len(self._columns),
        )

    def _sequence_tensors(self, *, samples: Sequence[CutoffSample]) -> SequenceTensors:
        series = [
            self._series_builder.build(
                user_ids=sample.dataset.meta["user_id"].to_numpy(dtype="int64"),
                feature_time=sample.cutoff.feature_time,
            )
            for sample in samples
        ]
        return (torch.from_numpy(np.concatenate(series)).to(self._device),)

    def _logits(self, *, sequences: SequenceTensors, features: torch.Tensor | None) -> torch.Tensor:
        return self._network(series=sequences[0], node_features=features)

    def _embedding(self, *, sequences: SequenceTensors) -> torch.Tensor:
        return self._network.embed(series=sequences[0])
