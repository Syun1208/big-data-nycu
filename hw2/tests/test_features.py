from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pandas.testing as pdt
import pytest

from src.data.classes.samples import FeatureContext
from src.data.classes.settings import FeatureSettings
from src.data.loaders.event_log import build_event_log
from src.data.loaders.relational_db import RelationalDatabase
from src.services.dataset_builder import FeatureAssembler
from src.services.factory import build_feature_blocks
from tests.factories import at

USERS = pd.Index([0, 1], name="user_id")
FUTURE_PUSH_DAYS = (7.5, 8.0, 20.0)


def features_at(*, database: RelationalDatabase, time: pd.Timestamp) -> pd.DataFrame:
    event_log = build_event_log(database=database)
    context = FeatureContext(
        user_ids=USERS,
        feature_time=time,
        known_events=event_log.known_before(time=time),
        database=database,
    )
    return FeatureAssembler(blocks=build_feature_blocks(settings=FeatureSettings())).build(context=context)


@pytest.fixture
def day_seven_features(database: RelationalDatabase) -> pd.DataFrame:
    return features_at(database=database, time=at(days=7))


@pytest.mark.parametrize(
    ("user", "column", "expected"),
    [
        pytest.param(0, "count_push_7d", 4, id="user0-count_push_7d"),
        pytest.param(1, "count_push_7d", 1, id="user1-count_push_7d"),
        pytest.param(0, "count_open_pr_7d", 1, id="user0-count_open_pr_7d"),
        pytest.param(0, "days_since_push", 1, id="user0-days_since_push"),
        pytest.param(0, "reviews_received_14d", 0, id="user0-reviews_received_14d"),
        pytest.param(0, "review_comments_received_14d", 1, id="user0-review_comments_received_14d"),
        pytest.param(0, "comments_received_14d", 1, id="user0-comments_received_14d"),
        pytest.param(0, "open_issues_assigned", 1, id="user0-open_issues_assigned"),
        pytest.param(0, "merged_prs_28d", 1, id="user0-merged_prs_28d"),
        pytest.param(0, "open_prs_reviewed", 1, id="user0-open_prs_reviewed"),
        pytest.param(0, "org_repo_share", 1.0, id="user0-org_repo_share"),
    ],
)
def test_activity_feature(day_seven_features: pd.DataFrame, user: int, column: str, expected: float) -> None:
    assert day_seven_features.loc[user, column] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("user", "column", "expected"),
    [
        pytest.param(0, "streak_push", 1, id="user0-streak_push"),
        pytest.param(0, "week_rate_push", 1 / 13, id="user0-week_rate_push"),
        pytest.param(0, "share_push_28d", 4 / 6, id="user0-share_push_28d"),
        pytest.param(1, "share_push_28d", 1 / 4, id="user1-share_push_28d"),
        pytest.param(0, "own_repo_push_share_28d", 0, id="user0-own_repo_push_share_28d"),
        pytest.param(0, "approve_share", 1, id="user0-approve_share"),
        pytest.param(0, "self_merge_share", 0, id="user0-self_merge_share"),
        pytest.param(1, "merges_done_13w", 1, id="user1-merges_done_13w"),
    ],
)
def test_behavior_feature(day_seven_features: pd.DataFrame, user: int, column: str, expected: float) -> None:
    assert day_seven_features.loc[user, column] == pytest.approx(expected)


def test_features_have_no_missing_values(day_seven_features: pd.DataFrame) -> None:
    assert not day_seven_features.isna().to_numpy().any()


def test_future_events_do_not_change_features(database_factory: Callable[..., RelationalDatabase]) -> None:
    base = features_at(database=database_factory(), time=at(days=7))
    future = features_at(database=database_factory(extra_push_days=FUTURE_PUSH_DAYS), time=at(days=7))
    pdt.assert_frame_equal(base, future)


@pytest.mark.parametrize(
    ("days", "column", "expected"),
    [
        pytest.param(5, "merged_prs_28d", 0, id="before-merge-not-merged"),
        pytest.param(5, "open_prs_authored", 1, id="before-merge-still-open"),
        pytest.param(7, "open_prs_authored", 0, id="after-merge-closed"),
    ],
)
def test_late_known_merge_is_hidden_before_merge_time(
    database: RelationalDatabase,
    days: float,
    column: str,
    expected: int,
) -> None:
    features = features_at(database=database, time=at(days=days))
    assert features.loc[0, column] == expected
