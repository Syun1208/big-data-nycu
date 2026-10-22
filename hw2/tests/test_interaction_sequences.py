from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from src.data.classes.settings import EVENTS, STHNSettings
from src.data.loaders.event_log import build_event_log
from src.data.loaders.relational_db import RelationalDatabase
from src.services.interaction_sequences import PAD_KIND, InteractionSequenceBuilder, InteractionSequences
from tests.factories import at

CODE = {event: code for code, event in enumerate(EVENTS)}
MAX_EDGES = 5
USER_IDS = np.array([0, 2, 1, 7])
ROW_OF_USER = {user: row for row, user in enumerate(USER_IDS)}
SECONDS_PER_DAY = 86400
FUTURE_PUSH_DAYS = (7.5, 20.0)


def sequences_at(*, database: RelationalDatabase, days: float) -> InteractionSequences:
    builder = InteractionSequenceBuilder(
        event_log=build_event_log(database=database),
        settings=STHNSettings(max_edges=MAX_EDGES, device="cpu"),
    )
    return builder.build(user_ids=USER_IDS, feature_time=at(days=days))


@pytest.fixture
def sequences(database: RelationalDatabase) -> InteractionSequences:
    return sequences_at(database=database, days=6.5)


@pytest.fixture
def base_sequences(database_factory: Callable[..., RelationalDatabase]) -> InteractionSequences:
    return sequences_at(database=database_factory(), days=7)


@pytest.fixture
def future_sequences(database_factory: Callable[..., RelationalDatabase]) -> InteractionSequences:
    return sequences_at(database=database_factory(extra_push_days=FUTURE_PUSH_DAYS), days=7)


def test_most_recent_events_come_first(sequences: InteractionSequences) -> None:
    expected = [CODE["push"], CODE["pr_review"], CODE["push"], CODE["open_pr"], CODE["push"]]
    assert sequences.kinds[ROW_OF_USER[0]].tolist() == expected


def test_event_ages_increase_along_the_sequence(sequences: InteractionSequences) -> None:
    assert np.all(np.diff(sequences.age_seconds[ROW_OF_USER[0]]) > 0)


def test_age_is_measured_from_the_feature_time(sequences: InteractionSequences) -> None:
    assert sequences.age_seconds[ROW_OF_USER[0], 0] == pytest.approx(0.5 * SECONDS_PER_DAY)


def test_short_history_is_padded(sequences: InteractionSequences) -> None:
    assert sequences.kinds[ROW_OF_USER[2]].tolist() == [CODE["open_pr"]] + [PAD_KIND] * (MAX_EDGES - 1)


@pytest.mark.parametrize(
    ("user", "expected"),
    [
        pytest.param(0, MAX_EDGES, id="user0-capped"),
        pytest.param(2, 1, id="user2-one-pr"),
        pytest.param(1, 4, id="user1-four-events"),
        pytest.param(7, 0, id="unknown-user"),
    ],
)
def test_sequence_length(sequences: InteractionSequences, user: int, expected: int) -> None:
    assert sequences.lengths[ROW_OF_USER[user]] == expected


def test_future_rows_do_not_change_event_kinds(base_sequences: InteractionSequences, future_sequences: InteractionSequences) -> None:
    np.testing.assert_array_equal(base_sequences.kinds, future_sequences.kinds)


def test_future_rows_do_not_change_event_ages(base_sequences: InteractionSequences, future_sequences: InteractionSequences) -> None:
    np.testing.assert_allclose(base_sequences.age_seconds, future_sequences.age_seconds)
