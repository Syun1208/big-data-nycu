from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from src.data.classes.settings import EVENTS
from src.services.graph_snapshot import NODE_FEATURE_DIMS, NODE_TYPES, NUM_EDGE_KINDS

DECAY_DAYS = (1.0, 3.0, 7.0, 28.0)
PERIOD_DAYS = (7.0,)
TIME_ENCODING_DIM = 1 + len(DECAY_DAYS) + 2 * len(PERIOD_DAYS)
EDGE_DIM = NUM_EDGE_KINDS + TIME_ENCODING_DIM
SAGE_CONV = "sage"
GAT_CONV = "gat"
TRANSFORMER_CONV = "transformer"
CONV_TYPES = (SAGE_CONV, GAT_CONV, TRANSFORMER_CONV)


@dataclass(frozen=True, slots=True)
class RelationTensors:
    source_type: str
    target_type: str
    source: torch.Tensor
    target: torch.Tensor
    edge_features: torch.Tensor


@dataclass(frozen=True, slots=True)
class GraphTensors:
    node_features: dict[str, torch.Tensor]
    relations: dict[str, RelationTensors]


def encode_edges(
    *,
    kind: torch.Tensor,
    age_days: torch.Tensor,
    lookback_days: float,
) -> torch.Tensor:
    age = age_days.clamp(min=0.0).unsqueeze(1)
    parts = [
        nn.functional.one_hot(kind, NUM_EDGE_KINDS).float(),
        (age / lookback_days).clamp(max=1.0),
    ]
    parts += [torch.exp(-age / decay) for decay in DECAY_DAYS]
    for period in PERIOD_DAYS:
        angle = 2 * math.pi * age / period
        parts += [torch.sin(angle), torch.cos(angle)]
    return torch.cat(parts, dim=1)


def relation_directions(*, relations: Sequence[str]) -> list[tuple[str, str, bool]]:
    directions = []
    for name in relations:
        directions.append((f"{name}__fwd", name, False))
        directions.append((f"{name}__rev", name, True))
    return directions


def scatter_softmax(
    *,
    scores: torch.Tensor,
    target: torch.Tensor,
    size: int,
) -> torch.Tensor:
    index = target.unsqueeze(1).expand_as(scores)
    maximum = torch.full((size, scores.shape[1]), float("-inf"), device=scores.device, dtype=scores.dtype)
    maximum = maximum.scatter_reduce(0, index, scores, reduce="amax", include_self=True)
    weights = torch.exp(scores - maximum[target])
    total = torch.zeros((size, scores.shape[1]), device=scores.device, dtype=scores.dtype).index_add_(0, target, weights)
    return weights / total[target].clamp(min=1e-12)


class EdgeAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        conv_type: str,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")
        self._heads = num_heads
        self._head_dim = hidden_dim // num_heads
        self._conv_type = conv_type
        if conv_type == GAT_CONV:
            self.target_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.score = nn.Parameter(torch.empty(num_heads, 2 * self._head_dim))
            nn.init.xavier_uniform_(self.score)
        else:
            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.key = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        *,
        message: torch.Tensor,
        target_hidden: torch.Tensor,
        target: torch.Tensor,
        size: int,
    ) -> torch.Tensor:
        heads, head_dim = self._heads, self._head_dim
        if self._conv_type == GAT_CONV:
            values = message.view(-1, heads, head_dim)
            anchor = self.target_proj(target_hidden)[target].view(-1, heads, head_dim)
            scores = nn.functional.leaky_relu((torch.cat([anchor, values], dim=-1) * self.score).sum(-1), 0.2)
        else:
            values = self.value(message).view(-1, heads, head_dim)
            keys = self.key(message).view(-1, heads, head_dim)
            queries = self.query(target_hidden)[target].view(-1, heads, head_dim)
            scores = (queries * keys).sum(-1) / head_dim**0.5
        weights = scatter_softmax(
            scores=scores,
            target=target,
            size=size,
        )
        weighted = (weights.unsqueeze(-1) * values).reshape(-1, heads * head_dim)
        return torch.zeros((size, heads * head_dim), device=message.device, dtype=message.dtype).index_add_(0, target, weighted)


class HeteroMessageLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        relations: Sequence[tuple[str, str, str]],
        dropout: float,
        conv_type: str = SAGE_CONV,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if conv_type not in CONV_TYPES:
            raise ValueError(f"Unknown conv_type {conv_type!r}. Use one of {CONV_TYPES}.")
        self._conv_type = conv_type
        self._directions = relation_directions(relations=[name for name, _, _ in relations])
        types = {name: (source, target) for name, source, target in relations}
        self._targets = {
            key: (types[name][0] if reverse else types[name][1])
            for key, name, reverse in self._directions
        }
        self.messages = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.Linear(hidden_dim + EDGE_DIM, hidden_dim),
                    nn.ReLU(),
                )
                for key, _, _ in self._directions
            }
        )
        if conv_type != SAGE_CONV:
            self.attention = nn.ModuleDict(
                {
                    key: EdgeAttention(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        conv_type=conv_type,
                    )
                    for key, _, _ in self._directions
                }
            )
        incoming = {node_type: [key for key, target in self._targets.items() if target == node_type] for node_type in NODE_TYPES}
        self._incoming = incoming
        self.updates = nn.ModuleDict(
            {
                node_type: nn.Linear(hidden_dim * (1 + len(keys)) + len(keys), hidden_dim)
                for node_type, keys in incoming.items()
            }
        )
        self.norms = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in NODE_TYPES})
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        *,
        hidden: dict[str, torch.Tensor],
        graph: GraphTensors,
    ) -> dict[str, torch.Tensor]:
        aggregated = {}
        for key, name, reverse in self._directions:
            relation = graph.relations[name]
            source = relation.target if reverse else relation.source
            target = relation.source if reverse else relation.target
            source_type = relation.target_type if reverse else relation.source_type
            target_type = self._targets[key]
            size = hidden[target_type].shape[0]
            message = self.messages[key](torch.cat([hidden[source_type][source], relation.edge_features], dim=1))
            total = torch.zeros((size, message.shape[1]), device=message.device, dtype=message.dtype).index_add_(0, target, message)
            degree = torch.zeros(size, device=message.device, dtype=message.dtype).index_add_(
                0, target, torch.ones_like(target, dtype=message.dtype)
            )
            if self._conv_type == SAGE_CONV:
                pooled = total / degree.clamp(min=1.0).unsqueeze(1)
            else:
                pooled = self.attention[key](
                    message=message,
                    target_hidden=hidden[target_type],
                    target=target,
                    size=size,
                )
            aggregated[key] = (pooled, torch.log1p(degree).unsqueeze(1))
        updated = {}
        for node_type, keys in self._incoming.items():
            parts = [hidden[node_type]]
            parts += [aggregated[key][0] for key in keys]
            parts += [aggregated[key][1] for key in keys]
            change = self.updates[node_type](torch.cat(parts, dim=1))
            updated[node_type] = self.norms[node_type](hidden[node_type] + self.dropout(torch.relu(change)))
        return updated


class RelationalGNN(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        relations: Sequence[tuple[str, str, str]],
        user_extra_dim: int,
        conv_type: str = SAGE_CONV,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        input_dims = dict(NODE_FEATURE_DIMS)
        input_dims["user"] += user_extra_dim
        self.encoders = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(input_dims[node_type], hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                )
                for node_type in NODE_TYPES
            }
        )
        self.layers = nn.ModuleList(
            [
                HeteroMessageLayer(
                    hidden_dim=hidden_dim,
                    relations=relations,
                    dropout=dropout,
                    conv_type=conv_type,
                    num_heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(EVENTS)),
        )

    def forward(
        self,
        *,
        graph: GraphTensors,
        user_index: torch.Tensor,
    ) -> torch.Tensor:
        hidden = {node_type: self.encoders[node_type](features) for node_type, features in graph.node_features.items()}
        for layer in self.layers:
            hidden = layer(
                hidden=hidden,
                graph=graph,
            )
        return self.head(hidden["user"][user_index])
