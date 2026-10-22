from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.classes.samples import FeatureContext
from src.data.classes.settings import EVENTS, FeatureSettings

SECONDS_PER_DAY = 86400.0
DAYS_PER_WEEK = 7


def event_age_days(
    *,
    events: pd.DataFrame,
    feature_time: pd.Timestamp,
) -> pd.Series:
    return (feature_time - events["event_time"]).dt.total_seconds() / SECONDS_PER_DAY


def pivot_by_event(
    *,
    keys: pd.DataFrame,
    values: pd.Series,
    aggregation: str,
    prefix: str,
    suffix: str,
    user_ids: pd.Index,
    fill_value: float,
) -> pd.DataFrame:
    grouped = (
        values.groupby(
            [keys["actor_id"], keys["event"]],
            observed=True,
        )
        .agg(aggregation)
        .unstack("event")
    )
    grouped.columns = [f"{prefix}{event}{suffix}" for event in grouped.columns.astype(str)]
    expected = [f"{prefix}{event}{suffix}" for event in EVENTS]
    return grouped.reindex(
        index=user_ids,
        columns=expected,
    ).fillna(fill_value)


class ActivityHistoryBlock:
    def __init__(self, *, settings: FeatureSettings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "activity_history"

    def build(self, *, context: FeatureContext) -> pd.DataFrame:
        user_ids = context.user_ids
        events = context.known_events[context.known_events["actor_id"].isin(user_ids)]
        age = event_age_days(
            events=events,
            feature_time=context.feature_time,
        )
        longest = max(self._settings.windows_days)
        recent_mask = age < longest
        recent = events[recent_mask]
        recent_age = age[recent_mask]

        blocks = []
        for window in self._settings.windows_days:
            blocks.append(
                pivot_by_event(
                    keys=recent,
                    values=(recent_age < window).astype("float32"),
                    aggregation="sum",
                    prefix="count_",
                    suffix=f"_{window}d",
                    user_ids=user_ids,
                    fill_value=0.0,
                )
            )
        for half_life in self._settings.half_lives_days:
            blocks.append(
                pivot_by_event(
                    keys=recent,
                    values=np.power(0.5, recent_age / half_life).astype("float32"),
                    aggregation="sum",
                    prefix="decay_",
                    suffix=f"_{half_life:g}d",
                    user_ids=user_ids,
                    fill_value=0.0,
                )
            )
        blocks.append(
            pivot_by_event(
                keys=events,
                values=age,
                aggregation="min",
                prefix="days_since_",
                suffix="",
                user_ids=user_ids,
                fill_value=self._settings.recency_cap_days,
            ).clip(upper=self._settings.recency_cap_days)
        )
        week_horizon = self._settings.active_weeks * DAYS_PER_WEEK
        week_mask = recent_age < week_horizon
        blocks.append(
            pivot_by_event(
                keys=recent[week_mask],
                values=np.floor(recent_age[week_mask] / DAYS_PER_WEEK),
                aggregation="nunique",
                prefix="active_weeks_",
                suffix=f"_{self._settings.active_weeks}w",
                user_ids=user_ids,
                fill_value=0.0,
            )
        )
        frame = pd.concat(blocks, axis=1)
        shortest, second = [window for window in sorted(self._settings.windows_days) if window >= DAYS_PER_WEEK][:2]
        for event in EVENTS:
            current = frame[f"count_{event}_{shortest}d"]
            previous = frame[f"count_{event}_{second}d"] - current
            frame[f"trend_{event}"] = current - previous
        return frame.join(
            self._overall_activity(
                events=events,
                age=age,
                user_ids=user_ids,
            )
        )

    def _overall_activity(
        self,
        *,
        events: pd.DataFrame,
        age: pd.Series,
        user_ids: pd.Index,
    ) -> pd.DataFrame:
        overall = pd.DataFrame(index=user_ids)
        actor = events["actor_id"]
        for window in self._settings.windows_days:
            overall[f"count_any_{window}d"] = (age < window).groupby(actor).sum().reindex(user_ids).fillna(0.0)
        month = age < self._settings.repo_lookback_days
        overall["active_days_28d"] = np.floor(age[month]).groupby(actor[month]).nunique().reindex(user_ids).fillna(0.0)
        overall["event_types_28d"] = events.loc[month, "event"].groupby(actor[month], observed=True).nunique().reindex(user_ids).fillna(0.0)
        overall["days_since_any"] = age.groupby(actor).min().reindex(user_ids).fillna(self._settings.recency_cap_days)
        overall["days_since_any"] = overall["days_since_any"].clip(upper=self._settings.recency_cap_days)
        return overall.astype("float32")


class ProfileBlock:
    def __init__(self, *, settings: FeatureSettings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "profile"

    def build(self, *, context: FeatureContext) -> pd.DataFrame:
        users = context.database.users.set_index("user_id").reindex(context.user_ids)
        first_seen = users["first_seen_at"].where(users["first_seen_at"] < context.feature_time)
        days_in_data = (context.feature_time - first_seen).dt.total_seconds() / SECONDS_PER_DAY
        frame = pd.DataFrame(index=context.user_ids)
        frame["days_in_data"] = days_in_data.fillna(0.0).astype("float32")
        return frame
