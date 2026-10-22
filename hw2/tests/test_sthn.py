from __future__ import annotations

import pytest
import torch

from src.data.classes.settings import EVENTS
from src.models.sthn import STHN
from src.services.interaction_sequences import PAD_KIND

BATCH = 4
MAX_EDGES = 10
KNOWN_EDGES = 3
HIDDEN_DIMS = 16
NODE_FEATURE_DIMS = 3
MAX_AGE_SECONDS = 1e6


def build_network(*, window_size: int) -> STHN:
    return STHN(
        max_edges=MAX_EDGES,
        window_size=window_size,
        time_dims=8,
        hidden_dims=HIDDEN_DIMS,
        predictor_dims=12,
        num_layers=1,
        num_heads=2,
        expansion_factor=2,
        dropout=0.0,
        node_feature_dims=NODE_FEATURE_DIMS,
    )


@pytest.fixture
def network() -> STHN:
    torch.manual_seed(0)
    return build_network(window_size=5).eval()


@pytest.fixture
def generator() -> torch.Generator:
    return torch.Generator().manual_seed(0)


@pytest.fixture
def kinds() -> torch.Tensor:
    values = torch.full((BATCH, MAX_EDGES), PAD_KIND, dtype=torch.int8)
    values[:, :KNOWN_EDGES] = torch.tensor([5, 0, 3], dtype=torch.int8)
    return values


@pytest.fixture
def ages(generator: torch.Generator) -> torch.Tensor:
    return torch.rand(BATCH, MAX_EDGES, generator=generator) * MAX_AGE_SECONDS


@pytest.fixture
def ages_with_noisy_padding(ages: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    noisy = ages.clone()
    noisy[:, KNOWN_EDGES:] = torch.rand(BATCH, MAX_EDGES - KNOWN_EDGES, generator=generator) * 10 * MAX_AGE_SECONDS
    return noisy


@pytest.fixture
def node_features() -> torch.Tensor:
    return torch.zeros(BATCH, NODE_FEATURE_DIMS)


def test_forward_returns_one_logit_per_event(
    network: STHN,
    kinds: torch.Tensor,
    ages: torch.Tensor,
    node_features: torch.Tensor,
) -> None:
    logits = network(kinds=kinds, age_seconds=ages, node_features=node_features)
    assert logits.shape == (BATCH, len(EVENTS))


def test_padded_positions_do_not_change_the_output(
    network: STHN,
    kinds: torch.Tensor,
    ages: torch.Tensor,
    ages_with_noisy_padding: torch.Tensor,
    node_features: torch.Tensor,
) -> None:
    torch.testing.assert_close(
        network(kinds=kinds, age_seconds=ages_with_noisy_padding, node_features=node_features),
        network(kinds=kinds, age_seconds=ages, node_features=node_features),
    )


def test_embedding_has_hidden_size(network: STHN, kinds: torch.Tensor, ages: torch.Tensor) -> None:
    assert network.embed(kinds=kinds, age_seconds=ages).shape == (BATCH, HIDDEN_DIMS)


def test_max_edges_must_split_into_windows() -> None:
    with pytest.raises(ValueError, match="divisible by window_size"):
        build_network(window_size=3)
