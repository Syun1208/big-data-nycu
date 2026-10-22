from __future__ import annotations

import pytest

from src.services.method_names import ensemble_name, ensemble_recipe, method_name


@pytest.fixture
def member_models() -> dict[str, str]:
    return {
        "xgboost_seed1": "xgboost",
        "xgboost_seed7": "xgboost",
        "gnn_mean": "rdl_gnn_hybrid",
        "gnn_gat": "rdl_gnn_hybrid_gat",
    }


@pytest.fixture
def weights() -> dict[str, float]:
    return {"xgboost_seed1": 0.2, "xgboost_seed7": 0.4, "gnn_mean": 0.0, "gnn_gat": 0.4}


def test_ensemble_name_lists_members_with_weight(weights: dict[str, float], member_models: dict[str, str]) -> None:
    assert ensemble_name(weights=weights, member_models=member_models) == "Ensemble (XGBoost + Hetero GAT)"


def test_ensemble_recipe_sums_weights_per_method(weights: dict[str, float], member_models: dict[str, str]) -> None:
    assert ensemble_recipe(weights=weights, member_models=member_models) == "0.6 XGBoost (2 runs) + 0.4 Hetero GAT"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        pytest.param("rdl_gnn_hybrid", "Hetero GNN", id="mean-aggregation"),
        pytest.param("rdl_gnn_hybrid_gat", "Hetero GAT", id="gat-aggregation"),
        pytest.param("rdl_gnn_hybrid_transformer", "Hetero Graph Transformer", id="transformer-aggregation"),
        pytest.param("unknown_model", "unknown_model", id="unknown-falls-back"),
    ],
)
def test_method_name(model: str, expected: str) -> None:
    assert method_name(model=model) == expected
