from __future__ import annotations

import pytest
import torch

from src.data.classes.settings import EVENTS
from src.models.stgnn import STGNN, SpatialGNN

BATCH = 4
NUM_STEPS = 5
HIDDEN_DIMS = 16
NODE_FEATURE_DIMS = 3


def build_network(*, node_feature_dims: int, num_heads: int = 4) -> STGNN:
    return STGNN(
        num_steps=NUM_STEPS,
        hidden_dims=HIDDEN_DIMS,
        num_heads=num_heads,
        num_layers=1,
        node_feature_dims=node_feature_dims,
    )


@pytest.fixture
def generator() -> torch.Generator:
    return torch.Generator().manual_seed(0)


@pytest.fixture
def series(generator: torch.Generator) -> torch.Tensor:
    return torch.rand(BATCH, NUM_STEPS, len(EVENTS), generator=generator)


@pytest.fixture
def network() -> STGNN:
    torch.manual_seed(0)
    return build_network(node_feature_dims=0).eval()


@pytest.fixture
def hybrid_network() -> STGNN:
    torch.manual_seed(0)
    return build_network(node_feature_dims=NODE_FEATURE_DIMS).eval()


def test_forward_returns_one_logit_per_event(network: STGNN, series: torch.Tensor) -> None:
    assert network(series=series, node_features=None).shape == (BATCH, len(EVENTS))


def test_hybrid_forward_returns_one_logit_per_event(hybrid_network: STGNN, series: torch.Tensor) -> None:
    logits = hybrid_network(series=series, node_features=torch.zeros(BATCH, NODE_FEATURE_DIMS))
    assert logits.shape == (BATCH, len(EVENTS))


def test_embedding_concatenates_event_node_states(network: STGNN, series: torch.Tensor) -> None:
    assert network.embed(series=series).shape == (BATCH, len(EVENTS) * HIDDEN_DIMS)


def test_spatial_gnn_keeps_node_shape(generator: torch.Generator) -> None:
    nodes = torch.rand(BATCH, len(EVENTS), HIDDEN_DIMS, generator=generator)
    assert SpatialGNN(dims=HIDDEN_DIMS)(nodes).shape == nodes.shape


def test_hidden_size_must_split_into_heads() -> None:
    with pytest.raises(ValueError, match="divisible by num_heads"):
        build_network(node_feature_dims=0, num_heads=3)
