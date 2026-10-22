from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from src.data.classes.settings import GnnSettings
from src.data.loaders.event_log import build_event_log
from src.data.loaders.relational_db import RelationalDatabase
from src.models.gnn_model import RELATIONS
from src.services.graph_snapshot import AUTHOR_KIND, NODE_FEATURE_DIMS, GraphSnapshot, GraphSnapshotBuilder
from tests.factories import at

RELATION_NAMES = [name for name, _, _ in RELATIONS]
PR_OPEN_COLUMN = 2
PR_MERGED_COLUMN = 3
FUTURE_PUSH_DAYS = (7.5, 20.0)


def snapshot_at(*, database: RelationalDatabase, days: float) -> GraphSnapshot:
    builder = GraphSnapshotBuilder(
        database=database,
        event_log=build_event_log(database=database),
        settings=GnnSettings(device="cpu"),
    )
    return builder.build(feature_time=at(days=days))


@pytest.fixture
def early_snapshot(database: RelationalDatabase) -> GraphSnapshot:
    return snapshot_at(database=database, days=4.5)


@pytest.fixture
def base_snapshot(database_factory: Callable[..., RelationalDatabase]) -> GraphSnapshot:
    return snapshot_at(database=database_factory(), days=7)


@pytest.fixture
def future_snapshot(database_factory: Callable[..., RelationalDatabase]) -> GraphSnapshot:
    return snapshot_at(database=database_factory(extra_push_days=FUTURE_PUSH_DAYS), days=7)


def test_user_repo_edges_only_include_known_events(early_snapshot: GraphSnapshot) -> None:
    assert early_snapshot.relations["user__repo"].size == 6


def test_edge_ages_are_positive(early_snapshot: GraphSnapshot) -> None:
    assert (early_snapshot.relations["user__repo"].age_days > 0).all()


def test_user_pr_edges_point_to_known_prs(early_snapshot: GraphSnapshot) -> None:
    assert set(early_snapshot.relations["user__pr"].target.tolist()) == {0, 1}


def test_pr_author_edges_have_author_kind(early_snapshot: GraphSnapshot) -> None:
    assert AUTHOR_KIND in early_snapshot.relations["user__pr"].kind.tolist()


def test_issue_of_a_pr_is_linked_to_the_pr(early_snapshot: GraphSnapshot) -> None:
    assert early_snapshot.relations["issue__pr"].size == 1


@pytest.mark.parametrize("relation", RELATION_NAMES)
def test_future_rows_do_not_change_edges(base_snapshot: GraphSnapshot, future_snapshot: GraphSnapshot, relation: str) -> None:
    np.testing.assert_array_equal(base_snapshot.relations[relation].source, future_snapshot.relations[relation].source)
    np.testing.assert_allclose(base_snapshot.relations[relation].age_days, future_snapshot.relations[relation].age_days)


@pytest.mark.parametrize(
    ("days", "column", "expected"),
    [
        pytest.param(5, PR_OPEN_COLUMN, 1.0, id="before-merge-open"),
        pytest.param(5, PR_MERGED_COLUMN, 0.0, id="before-merge-not-merged"),
        pytest.param(7, PR_MERGED_COLUMN, 1.0, id="after-merge-merged"),
    ],
)
def test_late_known_merge_flag(database: RelationalDatabase, days: float, column: int, expected: float) -> None:
    snapshot = snapshot_at(database=database, days=days)
    assert snapshot.node_features["pr"][0, column] == expected


def test_node_feature_widths(base_snapshot: GraphSnapshot) -> None:
    assert {name: values.shape[1] for name, values in base_snapshot.node_features.items()} == NODE_FEATURE_DIMS
