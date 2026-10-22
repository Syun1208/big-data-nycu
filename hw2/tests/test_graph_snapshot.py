from __future__ import annotations

import numpy as np
import pytest

from src.data.classes.settings import EVENTS, GnnSettings
from src.data.loaders.event_log import build_event_log
from src.services.graph_snapshot import AUTHOR_KIND, NODE_FEATURE_DIMS, GraphSnapshotBuilder
from tests.conftest import at, build_database


def snapshot_at(*, database, days: float):
    builder = GraphSnapshotBuilder(
        database=database,
        event_log=build_event_log(database=database),
        settings=GnnSettings(device="cpu"),
    )
    return builder, builder.build(feature_time=at(days=days))


def test_edges_only_use_known_rows(database) -> None:
    _, snapshot = snapshot_at(database=database, days=4.5)
    assert snapshot.relations["user__repo"].size == 6
    assert (snapshot.relations["user__repo"].age_days > 0).all()
    pr_edges = snapshot.relations["user__pr"]
    assert set(pr_edges.target.tolist()) == {0, 1}
    assert AUTHOR_KIND in pr_edges.kind.tolist()
    assert snapshot.relations["issue__pr"].size == 1


def test_future_rows_do_not_change_snapshot() -> None:
    _, base = snapshot_at(database=build_database(), days=7)
    _, future = snapshot_at(database=build_database(extra_push_days=(7.5, 20.0)), days=7)
    for name, edges in base.relations.items():
        np.testing.assert_array_equal(edges.source, future.relations[name].source)
        np.testing.assert_allclose(edges.age_days, future.relations[name].age_days)


def test_late_known_merge_flag(database) -> None:
    _, before = snapshot_at(database=database, days=5)
    _, after = snapshot_at(database=database, days=7)
    assert before.node_features["pr"][0, 2] == 1.0
    assert before.node_features["pr"][0, 3] == 0.0
    assert after.node_features["pr"][0, 3] == 1.0
    assert {name: values.shape[1] for name, values in after.node_features.items()} == NODE_FEATURE_DIMS


def test_scatter_softmax_normalizes_per_target() -> None:
    torch = pytest.importorskip("torch")
    from src.models.relational_gnn import scatter_softmax

    target = torch.tensor([0, 0, 2, 2, 2])
    weights = scatter_softmax(scores=torch.randn(5, 3), target=target, size=4)
    totals = torch.zeros(4, 3).index_add_(0, target, weights)
    assert torch.allclose(totals[[0, 2]], torch.ones(2, 3))
    assert torch.all(totals[[1, 3]] == 0)


@pytest.mark.parametrize("conv_type", ["sage", "gat", "transformer"])
def test_gnn_forward_shapes(database, conv_type) -> None:
    torch = pytest.importorskip("torch")
    from src.models.gnn_model import RELATIONS
    from src.models.relational_gnn import GraphTensors, RelationalGNN, RelationTensors, encode_edges

    _, snapshot = snapshot_at(database=database, days=7)
    graph = GraphTensors(
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
                    lookback_days=91.0,
                ),
            )
            for name, edges in snapshot.relations.items()
        },
    )
    network = RelationalGNN(hidden_dim=8, num_layers=2, dropout=0.0, relations=RELATIONS, user_extra_dim=0, conv_type=conv_type, num_heads=2)
    logits = network(graph=graph, user_index=torch.tensor([0, 1]))
    assert logits.shape == (2, len(EVENTS))
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in network.head.parameters())
