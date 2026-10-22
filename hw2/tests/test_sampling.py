from __future__ import annotations

import pandas as pd

from src.data.classes.samples import CALIBRATION_SPLIT, HOLDOUT_SPLIT, TRAIN_SPLIT
from src.data.classes.settings import EVENTS, SamplingSettings
from src.data.loaders.event_log import build_event_log
from src.services.sampling import LabelBuilder, PopulationSelector, SamplePlanner
from tests.conftest import at


def test_training_cutoffs_are_weekly_and_ordered() -> None:
    planner = SamplePlanner(settings=SamplingSettings())
    cutoffs = planner.plan_training_cutoffs(data_end=pd.Timestamp("2026-10-21 12:00:00"))
    times = [cutoff.time for cutoff in cutoffs]
    assert all(later - earlier == pd.Timedelta(weeks=1) for earlier, later in zip(times, times[1:]))
    assert all(time.dayofweek == 2 and time.hour == 16 for time in times)
    assert times[-1] + pd.Timedelta(days=7) <= pd.Timestamp("2026-10-21 12:00:00")
    assert [cutoff.split for cutoff in cutoffs[-2:]] == [CALIBRATION_SPLIT, HOLDOUT_SPLIT]
    assert {cutoff.split for cutoff in cutoffs[:-2]} == {TRAIN_SPLIT}
    assert all(cutoff.feature_time < cutoff.time and cutoff.population_time < cutoff.feature_time for cutoff in cutoffs)


def test_prediction_cutoff_matches_test_population_time() -> None:
    planner = SamplePlanner(settings=SamplingSettings())
    cutoff = planner.prediction_cutoff(data_end=pd.Timestamp("2026-11-11 12:00:00"))
    assert cutoff.feature_time == pd.Timestamp("2026-11-11 12:00:00")
    assert cutoff.population_time == pd.Timestamp("2026-10-28 12:00:00")


def test_population_and_labels(database) -> None:
    settings = SamplingSettings(population_min_records=5, population_lag_days=0, feature_gap_hours=0)
    event_log = build_event_log(database=database)
    cutoff = SamplePlanner(settings=settings).make_cutoff(time=at(days=7), split=TRAIN_SPLIT)
    users = PopulationSelector(settings=settings).select(
        event_log=event_log,
        cohort_user_ids=database.cohort_user_ids,
        cutoff=cutoff,
    )
    assert list(users) == [0]
    labels = LabelBuilder(settings=settings).build(event_log=event_log, user_ids=pd.Index([0, 1]), cutoff=cutoff)
    assert list(labels.columns) == list(EVENTS)
    assert labels.loc[0, "push"] == 1
    assert labels.loc[0, "issue_comment"] == 1
    assert labels.loc[1, "push"] == 1
    assert labels.loc[1, "open_pr"] == 0
