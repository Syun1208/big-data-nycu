from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from src.data.classes.samples import FeatureContext
from src.data.classes.settings import FeatureSettings
from src.data.loaders.event_log import build_event_log
from src.services.dataset_builder import FeatureAssembler
from src.services.factory import build_feature_blocks
from tests.conftest import at, build_database

USERS = pd.Index([0, 1], name="user_id")


def features_at(*, database, time: pd.Timestamp) -> pd.DataFrame:
    event_log = build_event_log(database=database)
    context = FeatureContext(
        user_ids=USERS,
        feature_time=time,
        known_events=event_log.known_before(time=time),
        database=database,
    )
    return FeatureAssembler(blocks=build_feature_blocks(settings=FeatureSettings())).build(context=context)


def test_activity_counts(database) -> None:
    features = features_at(database=database, time=at(days=7))
    assert features.loc[0, "count_push_7d"] == 4
    assert features.loc[1, "count_push_7d"] == 1
    assert features.loc[0, "count_open_pr_7d"] == 1
    assert features.loc[0, "days_since_push"] == 1
    assert features.loc[0, "reviews_received_14d"] == 0
    assert features.loc[0, "review_comments_received_14d"] == 1
    assert features.loc[0, "comments_received_14d"] == 1
    assert features.loc[0, "open_issues_assigned"] == 1
    assert features.loc[0, "merged_prs_28d"] == 1
    assert features.loc[0, "open_prs_reviewed"] == 1
    assert features.loc[0, "org_repo_share"] == 1.0
    assert features.notna().all().all()


def test_future_events_do_not_change_features() -> None:
    time = at(days=7)
    base = features_at(database=build_database(), time=time)
    future = features_at(database=build_database(extra_push_days=(7.5, 8.0, 20.0)), time=time)
    pdt.assert_frame_equal(base, future)


def test_late_known_merge_is_hidden_before_merge_time(database) -> None:
    before = features_at(database=database, time=at(days=5))
    after = features_at(database=database, time=at(days=7))
    assert before.loc[0, "merged_prs_28d"] == 0
    assert before.loc[0, "open_prs_authored"] == 1
    assert after.loc[0, "open_prs_authored"] == 0


def test_behavior_features(database) -> None:
    features = features_at(database=database, time=at(days=7))
    assert features.loc[0, "streak_push"] == 1
    assert 0 < features.loc[0, "week_rate_push"] <= 1
    assert features.loc[0, "share_push_28d"] == features.loc[0, "count_push_28d"] / features.loc[0, "count_any_28d"]
    assert features.loc[0, "own_repo_push_share_28d"] == 0
    assert features.loc[0, "approve_share"] == 1
    assert features.loc[0, "self_merge_share"] == 0
    assert features.loc[1, "merges_done_13w"] == 1
    assert features.notna().all().all()
