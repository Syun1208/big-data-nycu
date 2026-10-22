from __future__ import annotations

import pytest
import torch

from src.data.classes.settings import EVENTS, GnnSettings
from src.data.loaders.event_log import build_event_log
from src.data.loaders.relational_db import RelationalDatabase
from src.models.gnn_model import RELATIONS
from src.models.relational_gnn import GraphTensors, RelationalGNN, RelationTensors, encode_edges, scatter_softmax
from src.services.graph_snapshot import GraphSnapshotBuilder
from tests.factories import at

CONV_TYPES = ["sage", "gat", "transformer"]
LOOKBACK_DAYS = 91.0
USER_INDEX = [0, 1]
NUM_TARGETS = 4
EMPTY_TARGETS = [1, 3]
FILLED_TARGETS = [0, 2]


@pytest.fixture
def graph(database: RelationalDatabase) -> GraphTensors:
    snapshot = GraphSnapshotBuilder(
        database=database,
        event_log=build_event_log(database=database),
        settings=GnnSettings(device="cpu"),
    ).build(feature_time=at(days=7))
    return GraphTensors(
        node_features={name: torch.from_numpy(values) for name, values in snapshot.node_features.items()},
        relations={
            name: RelationTensors(
                source_type=edges.source_type,
                target_type=edges.target_type,
                source=torch.from_numpy(edges.source),
                target=torch.from_numpy(edges.target),
                edge_features=encode_edges(
                    kind=torch.from_numpy(edges.kind),
                    age_days=torch.from_numpy(edges.age_days),
                    lookback_days=LOOKBACK_DAYS,
                ),
            )
            for name, edges in snapshot.relations.items()
        },
    )


@pytest.fixture(params=CONV_TYPES)
def network(request: pytest.FixtureRequest) -> RelationalGNN:
    torch.manual_seed(0)
    return RelationalGNN(
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        relations=RELATIONS,
        user_extra_dim=0,
        conv_type=request.param,
        num_heads=2,
    )


@pytest.fixture
def softmax_weights() -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.tensor([0, 0, 2, 2, 2])
    scores = torch.randn(5, 3, generator=torch.Generator().manual_seed(0))
    return target, scatter_softmax(scores=scores, target=target, size=NUM_TARGETS)


@pytest.fixture
def softmax_totals(softmax_weights: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    target, weights = softmax_weights
    return torch.zeros(NUM_TARGETS, 3).index_add_(0, target, weights)


def test_scatter_softmax_sums_to_one_per_target(softmax_totals: torch.Tensor) -> None:
    torch.testing.assert_close(softmax_totals[FILLED_TARGETS], torch.ones(len(FILLED_TARGETS), 3))


def test_scatter_softmax_is_zero_for_targets_without_edges(softmax_totals: torch.Tensor) -> None:
    torch.testing.assert_close(softmax_totals[EMPTY_TARGETS], torch.zeros(len(EMPTY_TARGETS), 3))


def test_forward_returns_one_logit_per_event(network: RelationalGNN, graph: GraphTensors) -> None:
    logits = network(graph=graph, user_index=torch.tensor(USER_INDEX))
    assert logits.shape == (len(USER_INDEX), len(EVENTS))


def test_backward_reaches_the_prediction_head(network: RelationalGNN, graph: GraphTensors) -> None:
    network(graph=graph, user_index=torch.tensor(USER_INDEX)).sum().backward()
    assert all(parameter.grad is not None for parameter in network.head.parameters())
