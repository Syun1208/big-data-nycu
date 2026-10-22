from __future__ import annotations

from collections.abc import Mapping

ENSEMBLE_MODEL = "ensemble"
METHOD_NAMES = {
    "xgboost": "XGBoost",
    "rdl_gnn": "Hetero GNN",
    "rdl_gnn_hybrid": "Hetero GNN",
    "rdl_gnn_hybrid_gat": "Hetero GAT",
    "rdl_gnn_hybrid_transformer": "Hetero Graph Transformer",
    "sthn": "STHN",
    "sthn_hybrid": "STHN",
    "stgnn": "STGNN",
    "stgnn_hybrid": "STGNN",
    "tabpfn_v2": "TabPFN v2",
    ENSEMBLE_MODEL: "Ensemble",
}


def method_name(*, model: str) -> str:
    return METHOD_NAMES.get(model, model)


def member_weights(*, weights: Mapping[str, float], member_models: Mapping[str, str]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for run, weight in weights.items():
        if weight > 0:
            name = method_name(model=member_models[run])
            totals[name] = totals.get(name, 0.0) + weight
    return dict(sorted(totals.items(), key=lambda item: -item[1]))


def ensemble_name(*, weights: Mapping[str, float], member_models: Mapping[str, str]) -> str:
    names = member_weights(weights=weights, member_models=member_models)
    return f"Ensemble ({' + '.join(names)})"


def ensemble_recipe(*, weights: Mapping[str, float], member_models: Mapping[str, str]) -> str:
    runs = {}
    for run, weight in weights.items():
        if weight > 0:
            name = method_name(model=member_models[run])
            runs[name] = runs.get(name, 0) + 1
    names = member_weights(weights=weights, member_models=member_models)
    parts = [f"{weight:.1f} {name}" + (f" ({runs[name]} runs)" if runs[name] > 1 else "") for name, weight in names.items()]
    return " + ".join(parts)
