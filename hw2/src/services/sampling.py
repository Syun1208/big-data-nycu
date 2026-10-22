from __future__ import annotations

import logging

import pandas as pd

from src.data.classes.samples import CALIBRATION_SPLIT, HOLDOUT_SPLIT, PREDICT_SPLIT, TRAIN_SPLIT, Cutoff
from src.data.classes.settings import EVENTS, SamplingSettings
from src.data.loaders.event_log import EventLog

logger = logging.getLogger(__name__)

WEEK = pd.Timedelta(weeks=1)


class SamplePlanner:
    def __init__(self, *, settings: SamplingSettings) -> None:
        self._settings = settings

    def make_cutoff(
        self,
        *,
        time: pd.Timestamp,
        split: str,
        data_end: pd.Timestamp | None = None,
    ) -> Cutoff:
        feature_time = time - self._settings.feature_gap
        if data_end is not None and feature_time > data_end:
            logger.warning(
                "Data ends at %s, before the feature time %s of cutoff %s. Features use the data available now.",
                data_end,
                feature_time,
                time,
            )
            feature_time = data_end
        population_time = time - pd.Timedelta(days=self._settings.population_lag_days) - self._settings.feature_gap
        return Cutoff(
            time=time,
            feature_time=feature_time,
            population_time=population_time,
            split=split,
        )

    def plan_training_cutoffs(self, *, data_end: pd.Timestamp) -> list[Cutoff]:
        times = []
        time = self._settings.anchor
        while time >= self._settings.start:
            if time + self._settings.horizon <= data_end:
                times.append(time)
            time = time - WEEK
        times = sorted(times)
        reserved = self._settings.calibration_weeks + self._settings.holdout_weeks
        if len(times) <= reserved:
            raise ValueError(f"Only {len(times)} complete weeks between {self._settings.start} and {data_end}")
        splits = (
            [TRAIN_SPLIT] * (len(times) - reserved)
            + [CALIBRATION_SPLIT] * self._settings.calibration_weeks
            + [HOLDOUT_SPLIT] * self._settings.holdout_weeks
        )
        cutoffs = [
            self.make_cutoff(
                time=time,
                split=split,
            )
            for time, split in zip(times, splits)
        ]
        for split in (TRAIN_SPLIT, CALIBRATION_SPLIT, HOLDOUT_SPLIT):
            chosen = [cutoff.label for cutoff in cutoffs if cutoff.split == split]
            logger.info("%s cutoffs (%d): %s to %s", split, len(chosen), chosen[0], chosen[-1])
        return cutoffs

    def prediction_cutoff(self, *, data_end: pd.Timestamp) -> Cutoff:
        return self.make_cutoff(
            time=self._settings.anchor,
            split=PREDICT_SPLIT,
            data_end=data_end,
        )


class PopulationSelector:
    def __init__(self, *, settings: SamplingSettings) -> None:
        self._settings = settings

    def select(
        self,
        *,
        event_log: EventLog,
        cohort_user_ids: pd.Index,
        cutoff: Cutoff,
    ) -> pd.Index:
        window_start = cutoff.population_time - pd.Timedelta(days=self._settings.population_window_days)
        known = event_log.known_before(time=cutoff.population_time)
        recent = known[(known["event_time"] >= window_start) & known["actor_id"].isin(cohort_user_ids)]
        counts = recent.groupby("actor_id").size()
        selected = counts.index[counts >= self._settings.population_min_records]
        return pd.Index(selected.sort_values(), name="user_id")


class LabelBuilder:
    def __init__(self, *, settings: SamplingSettings) -> None:
        self._settings = settings

    def build(
        self,
        *,
        event_log: EventLog,
        user_ids: pd.Index,
        cutoff: Cutoff,
    ) -> pd.DataFrame:
        window = event_log.occurring_between(
            start=cutoff.time,
            end=cutoff.time + self._settings.horizon,
        )
        window = window[window["actor_id"].isin(user_ids)]
        happened = (
            window.groupby(["actor_id", "event"], observed=True).size().unstack("event", fill_value=0).gt(0).astype("int8")
        )
        happened.columns = happened.columns.astype(str)
        labels = happened.reindex(
            index=user_ids,
            columns=list(EVENTS),
            fill_value=0,
        )
        labels.index.name = "user_id"
        return labels.astype("int8")
