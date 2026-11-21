from __future__ import annotations

import pandas as pd
import pytest

from src.data.classes.samples import CALIBRATION_SPLIT, HOLDOUT_SPLIT, TRAIN_SPLIT, Cutoff
from src.data.classes.settings import EVENTS, SamplingSettings
from src.data.loaders.event_log import EventLog
from src.data.loaders.relational_db import RelationalDatabase
from src.services.sampling import LabelBuilder, PopulationSelector, SamplePlanner
from tests.factories import at

DATA_END = pd.Timestamp("2026-10-21 12:00:00")
TEST_DATA_END = pd.Timestamp("2026-11-21 12:00:00")
TEST_POPULATION_TIME = pd.Timestamp("2026-11-07 12:00:00")
LATER_DATA_END = pd.Timestamp("2026-10-30 12:00:00")
PINNED_VALIDATION = pd.Timestamp("2026-10-03 16:00:00")
PINNED_CALIBRATION = pd.Timestamp("2026-09-26 16:00:00")
ONE_WEEK = pd.Timedelta(weeks=1)
SATURDAY = 5
CUTOFF_HOUR_UTC = 16


@pytest.fixture
def planner() -> SamplePlanner:
    return SamplePlanner(settings=SamplingSettings())


@pytest.fixture
def training_cutoffs(planner: SamplePlanner) -> list[Cutoff]:
    return planner.plan_training_cutoffs(data_end=DATA_END)


@pytest.fixture
def cutoff_times(training_cutoffs: list[Cutoff]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([cutoff.time for cutoff in training_cutoffs])


@pytest.fixture
def prediction_cutoff(planner: SamplePlanner) -> Cutoff:
    return planner.prediction_cutoff(data_end=TEST_DATA_END)


@pytest.fixture
def pinned_cutoffs() -> list[Cutoff]:
    settings = SamplingSettings(last_validation_cutoff=str(PINNED_VALIDATION))
    return SamplePlanner(settings=settings).plan_training_cutoffs(data_end=LATER_DATA_END)


@pytest.fixture
def small_settings() -> SamplingSettings:
    return SamplingSettings(
        population_min_records=5,
        population_lag_days=0,
        feature_gap_hours=0,
    )


@pytest.fixture
def day_seven_cutoff(small_settings: SamplingSettings) -> Cutoff:
    return SamplePlanner(settings=small_settings).make_cutoff(time=at(days=7), split=TRAIN_SPLIT)


@pytest.fixture
def labels(small_settings: SamplingSettings, event_log: EventLog, day_seven_cutoff: Cutoff) -> pd.DataFrame:
    return LabelBuilder(settings=small_settings).build(
        event_log=event_log,
        user_ids=pd.Index([0, 1]),
        cutoff=day_seven_cutoff,
    )


def test_training_cutoffs_are_one_week_apart(cutoff_times: pd.DatetimeIndex) -> None:
    assert set(cutoff_times.to_series().diff().dropna()) == {ONE_WEEK}


def test_training_cutoffs_fall_on_saturday_afternoon_utc(cutoff_times: pd.DatetimeIndex) -> None:
    assert set(cutoff_times.dayofweek) == {SATURDAY}
    assert set(cutoff_times.hour) == {CUTOFF_HOUR_UTC}


def test_last_label_week_ends_before_data_end(cutoff_times: pd.DatetimeIndex) -> None:
    assert cutoff_times[-1] + ONE_WEEK <= DATA_END


def test_last_two_cutoffs_are_calibration_then_validation(training_cutoffs: list[Cutoff]) -> None:
    assert [cutoff.split for cutoff in training_cutoffs[-2:]] == [CALIBRATION_SPLIT, HOLDOUT_SPLIT]


def test_earlier_cutoffs_are_train(training_cutoffs: list[Cutoff]) -> None:
    assert {cutoff.split for cutoff in training_cutoffs[:-2]} == {TRAIN_SPLIT}


def test_population_and_features_are_taken_before_the_cutoff(training_cutoffs: list[Cutoff]) -> None:
    assert all(cutoff.population_time < cutoff.feature_time < cutoff.time for cutoff in training_cutoffs)


def test_pinned_validation_week_ignores_newer_data(pinned_cutoffs: list[Cutoff]) -> None:
    assert (pinned_cutoffs[-1].time, pinned_cutoffs[-1].split) == (PINNED_VALIDATION, HOLDOUT_SPLIT)


def test_calibration_is_the_week_before_the_pinned_validation(pinned_cutoffs: list[Cutoff]) -> None:
    assert (pinned_cutoffs[-2].time, pinned_cutoffs[-2].split) == (PINNED_CALIBRATION, CALIBRATION_SPLIT)


def test_prediction_features_use_the_last_data(prediction_cutoff: Cutoff) -> None:
    assert prediction_cutoff.feature_time == TEST_DATA_END


def test_prediction_population_matches_test_input_time(prediction_cutoff: Cutoff) -> None:
    assert prediction_cutoff.population_time == TEST_POPULATION_TIME


def test_population_keeps_cohort_users_with_enough_records(
    small_settings: SamplingSettings,
    event_log: EventLog,
    database: RelationalDatabase,
    day_seven_cutoff: Cutoff,
) -> None:
    users = PopulationSelector(settings=small_settings).select(
        event_log=event_log,
        cohort_user_ids=database.cohort_user_ids,
        cutoff=day_seven_cutoff,
    )
    assert list(users) == [0]


def test_label_columns_follow_event_order(labels: pd.DataFrame) -> None:
    assert list(labels.columns) == list(EVENTS)


@pytest.mark.parametrize(
    ("user", "event", "expected"),
    [
        pytest.param(0, "push", 1, id="user0-push"),
        pytest.param(0, "issue_comment", 1, id="user0-issue_comment"),
        pytest.param(1, "push", 1, id="user1-push"),
        pytest.param(1, "open_pr", 0, id="user1-open_pr"),
    ],
)
def test_next_week_label(labels: pd.DataFrame, user: int, event: str, expected: int) -> None:
    assert labels.loc[user, event] == expected
