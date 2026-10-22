from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.classes.samples import FeatureContext
from src.data.classes.settings import EVENTS, FeatureSettings
from src.services.temporal_features import DAYS_PER_WEEK, event_age_days

HOUR_BINS = ((0, 6), (6, 12), (12, 18), (18, 24))
WEEKEND_DAYS = (5, 6)
SMOOTHING = 0.5


def safe_ratio(
    *,
    numerator: np.ndarray | pd.Series,
    denominator: np.ndarray | pd.Series,
) -> np.ndarray:
    numerator = np.asarray(numerator, dtype="float64")
    denominator = np.asarray(denominator, dtype="float64")
    return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), 0.0)


def weekly_presence(
    *,
    rows: np.ndarray,
    events: np.ndarray,
    weeks: np.ndarray,
    num_users: int,
    num_weeks: int,
) -> np.ndarray:
    presence = np.zeros((num_users, len(EVENTS), num_weeks), dtype=bool)
    presence[rows, events, weeks] = True
    return presence


def current_streak(*, presence: np.ndarray) -> np.ndarray:
    inactive = ~presence
    first_gap = np.where(inactive.any(axis=-1), inactive.argmax(axis=-1), presence.shape[-1])
    return first_gap.astype("float32")


def transition_rates(*, presence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    newer = presence[..., :-1]
    older = presence[..., 1:]
    rate = presence.mean(axis=-1)
    stay = (np.sum(newer & older, axis=-1) + SMOOTHING * rate) / (np.sum(older, axis=-1) + SMOOTHING)
    back = (np.sum(newer & ~older, axis=-1) + SMOOTHING * rate) / (np.sum(~older, axis=-1) + SMOOTHING)
    return stay.astype("float32"), back.astype("float32")


class BehaviorBlock:
    def __init__(self, *, settings: FeatureSettings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "behavior"

    def build(self, *, context: FeatureContext) -> pd.DataFrame:
        user_ids = context.user_ids
        events = context.known_events[context.known_events["actor_id"].isin(user_ids)]
        age = event_age_days(
            events=events,
            feature_time=context.feature_time,
        )
        horizon = self._settings.behavior_weeks * DAYS_PER_WEEK
        window = events[age < horizon]
        window_age = age[age < horizon].to_numpy()
        rows = pd.Index(user_ids).get_indexer(window["actor_id"])
        codes = window["event"].cat.codes.to_numpy()

        frame = pd.DataFrame(index=user_ids)
        frame = frame.join(self._weekly_habits(rows=rows, codes=codes, age=window_age, user_ids=user_ids))
        frame = frame.join(self._rhythm(window=window, rows=rows, age=window_age, user_ids=user_ids))
        frame = frame.join(self._mix(window=window, age=window_age, user_ids=user_ids))
        frame = frame.join(self._ownership(context=context, window=window, age=window_age))
        return frame.astype("float32")

    def _weekly_habits(
        self,
        *,
        rows: np.ndarray,
        codes: np.ndarray,
        age: np.ndarray,
        user_ids: pd.Index,
    ) -> pd.DataFrame:
        num_weeks = self._settings.behavior_weeks
        weeks = np.minimum((age // DAYS_PER_WEEK).astype("int64"), num_weeks - 1)
        presence = weekly_presence(
            rows=rows,
            events=codes,
            weeks=weeks,
            num_users=len(user_ids),
            num_weeks=num_weeks,
        )
        any_presence = presence.any(axis=1)
        streak = current_streak(presence=presence)
        any_streak = current_streak(presence=any_presence)
        stay, back = transition_rates(presence=presence)
        any_stay, any_back = transition_rates(presence=any_presence)

        weekly_counts = np.zeros((len(user_ids), num_weeks), dtype="float64")
        np.add.at(weekly_counts, (rows, weeks), 1.0)
        mean_week = weekly_counts.mean(axis=1)

        habits = {}
        for index, event in enumerate(EVENTS):
            habits[f"week_rate_{event}"] = presence[:, index].mean(axis=1)
            habits[f"streak_{event}"] = streak[:, index]
            habits[f"stay_{event}"] = stay[:, index]
            habits[f"return_{event}"] = back[:, index]
        habits["week_rate_any"] = any_presence.mean(axis=1)
        habits["streak_any"] = any_streak
        habits["stay_any"] = any_stay
        habits["return_any"] = any_back
        habits["weekly_mean"] = mean_week
        habits["weekly_max"] = weekly_counts.max(axis=1)
        habits["weekly_cv"] = safe_ratio(numerator=weekly_counts.std(axis=1), denominator=mean_week)
        habits["last_week_vs_mean"] = safe_ratio(numerator=weekly_counts[:, 0], denominator=mean_week)
        return pd.DataFrame(habits, index=user_ids)

    def _rhythm(
        self,
        *,
        window: pd.DataFrame,
        rows: np.ndarray,
        age: np.ndarray,
        user_ids: pd.Index,
    ) -> pd.DataFrame:
        count = np.bincount(rows, minlength=len(user_ids)).astype("float64")
        hours = window["event_time"].dt.hour.to_numpy()
        weekdays = window["event_time"].dt.dayofweek.to_numpy()
        rhythm = {}
        for start, end in HOUR_BINS:
            inside = (hours >= start) & (hours < end)
            rhythm[f"hour_share_{start:02d}_{end:02d}"] = safe_ratio(
                numerator=np.bincount(rows[inside], minlength=len(user_ids)),
                denominator=count,
            )
        weekend = np.isin(weekdays, WEEKEND_DAYS)
        rhythm["weekend_share"] = safe_ratio(
            numerator=np.bincount(rows[weekend], minlength=len(user_ids)),
            denominator=count,
        )
        days = pd.DataFrame({"row": rows, "day": np.floor(age)}).drop_duplicates().sort_values(["row", "day"])
        gaps = days.groupby("row")["day"].diff().dropna()
        mean_gap = gaps.groupby(days.loc[gaps.index, "row"]).mean().reindex(range(len(user_ids))).to_numpy()
        last_day = days.groupby("row")["day"].min().reindex(range(len(user_ids))).to_numpy()
        cap = float(self._settings.behavior_weeks * DAYS_PER_WEEK)
        mean_gap = np.nan_to_num(mean_gap, nan=cap)
        last_day = np.nan_to_num(last_day, nan=cap)
        rhythm["mean_gap_days"] = mean_gap
        rhythm["overdue_ratio"] = safe_ratio(numerator=last_day, denominator=np.maximum(mean_gap, 1.0))
        rhythm["hours_active_28d"] = (
            pd.Series(window["event_time"].dt.floor("h").to_numpy()[age < self._settings.repo_lookback_days])
            .groupby(rows[age < self._settings.repo_lookback_days])
            .nunique()
            .reindex(range(len(user_ids)))
            .fillna(0.0)
            .to_numpy()
        )
        return pd.DataFrame(rhythm, index=user_ids)

    def _mix(
        self,
        *,
        window: pd.DataFrame,
        age: np.ndarray,
        user_ids: pd.Index,
    ) -> pd.DataFrame:
        month = window[age < self._settings.repo_lookback_days]
        counts = month.groupby(["actor_id", "event"], observed=True).size().unstack("event", fill_value=0)
        counts.columns = counts.columns.astype(str)
        counts = counts.reindex(index=user_ids, columns=list(EVENTS), fill_value=0)
        total = counts.sum(axis=1)
        mix = pd.DataFrame(index=user_ids)
        for event in EVENTS:
            mix[f"share_{event}_28d"] = safe_ratio(numerator=counts[event], denominator=total)
        repo_counts = month.dropna(subset=["repo_id"]).groupby(["actor_id", "repo_id"]).size()
        share = repo_counts / repo_counts.groupby(level="actor_id").transform("sum")
        entropy = (-(share * np.log(share))).groupby(level="actor_id").sum()
        mix["repo_entropy_28d"] = entropy.reindex(user_ids).fillna(0.0)
        return mix

    def _ownership(
        self,
        *,
        context: FeatureContext,
        window: pd.DataFrame,
        age: np.ndarray,
    ) -> pd.DataFrame:
        user_ids = context.user_ids
        database = context.database
        time = context.feature_time
        logins = database.users.set_index("user_id")["login"].str.lower()
        owners = database.repositories.set_index("repo_id")["name"].str.split("/").str[0].str.lower()

        pushes = window[(window["event"] == "push") & (age < self._settings.repo_lookback_days)].dropna(subset=["repo_id"])
        own = pushes["repo_id"].astype("int64").map(owners).to_numpy() == pushes["actor_id"].map(logins).to_numpy()
        ownership = pd.DataFrame(index=user_ids)
        ownership["own_repo_push_share_28d"] = safe_ratio(
            numerator=pd.Series(own, index=pushes.index).groupby(pushes["actor_id"]).sum().reindex(user_ids).fillna(0.0),
            denominator=pushes.groupby("actor_id").size().reindex(user_ids).fillna(0.0),
        )

        since = time - pd.Timedelta(days=self._settings.behavior_weeks * DAYS_PER_WEEK)
        reviews = database.pr_reviews
        reviews = reviews[
            (reviews["first_seen_at"] < time)
            & reviews["submitted_at"].between(since, time, inclusive="left")
            & reviews["reviewer_id"].isin(user_ids)
        ]
        ownership["approve_share"] = safe_ratio(
            numerator=(reviews["state"] == "approved").groupby(reviews["reviewer_id"]).sum().reindex(user_ids).fillna(0.0),
            denominator=reviews.groupby("reviewer_id").size().reindex(user_ids).fillna(0.0),
        )

        pull_requests = database.pull_requests
        merged = pull_requests[
            (pull_requests["first_seen_at"] < time)
            & pull_requests["merged_at"].notna()
            & pull_requests["merged_at"].between(since, time, inclusive="left")
            & pull_requests["author_id"].isin(user_ids)
        ]
        self_merged = merged["merged_by_id"].astype("Float64") == merged["author_id"].astype("Float64")
        ownership["self_merge_share"] = safe_ratio(
            numerator=self_merged.fillna(False).groupby(merged["author_id"]).sum().reindex(user_ids).fillna(0.0),
            denominator=merged.groupby("author_id").size().reindex(user_ids).fillna(0.0),
        )
        merges_done = pull_requests[
            (pull_requests["first_seen_at"] < time)
            & pull_requests["merged_at"].notna()
            & pull_requests["merged_at"].between(since, time, inclusive="left")
            & pull_requests["merged_by_id"].isin(user_ids)
        ]
        ownership["merges_done_13w"] = merges_done.groupby("merged_by_id").size().reindex(user_ids).fillna(0.0)
        return ownership
