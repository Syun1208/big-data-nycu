from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from src.data.classes.settings import EVENTS, STGNNSettings
from src.data.loaders.event_log import build_event_log
from src.data.loaders.relational_db import RelationalDatabase
from src.services.activity_series import ActivitySeriesBuilder
from tests.factories import at

CODE = {event: code for code, event in enumerate(EVENTS)}
NUM_STEPS = 3
STEP_DAYS = 3
USER_IDS = np.array([0, 1, 7])
ROW_OF_USER = {user: row for row, user in enumerate(USER_IDS)}
FUTURE_PUSH_DAYS = (7.5, 20.0)


def counts_at(*, database: RelationalDatabase, days: float) -> np.ndarray:
    builder = ActivitySeriesBuilder(
        event_log=build_event_log(database=database),
        settings=STGNNSettings(num_steps=NUM_STEPS, step_days=STEP_DAYS, device="cpu"),
    )
    return np.expm1(builder.build(user_ids=USER_IDS, feature_time=at(days=days)))


@pytest.fixture
def counts(database: RelationalDatabase) -> np.ndarray:
    return counts_at(database=database, days=7)


def test_series_shape_is_users_by_steps_by_events(counts: np.ndarray) -> None:
    assert counts.shape == (len(USER_IDS), NUM_STEPS, len(EVENTS))


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        pytest.param("push", [1, 2, 1], id="push"),
        pytest.param("open_pr", [0, 1, 0], id="open_pr"),
        pytest.param("pr_review", [0, 0, 1], id="pr_review"),
        pytest.param("issue_comment", [0, 0, 0], id="issue_comment-in-future"),
    ],
)
def test_weekly_counts_oldest_step_first(counts: np.ndarray, event: str, expected: list[int]) -> None:
    np.testing.assert_allclose(counts[ROW_OF_USER[0], :, CODE[event]], expected)


def test_unknown_user_has_no_activity(counts: np.ndarray) -> None:
    assert counts[ROW_OF_USER[7]].sum() == 0


def test_future_rows_do_not_change_series(database_factory: Callable[..., RelationalDatabase]) -> None:
    base = counts_at(database=database_factory(), days=7)
    future = counts_at(database=database_factory(extra_push_days=FUTURE_PUSH_DAYS), days=7)
    np.testing.assert_array_equal(base, future)
