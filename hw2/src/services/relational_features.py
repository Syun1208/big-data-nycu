from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.classes.samples import FeatureContext
from src.data.classes.settings import EVENTS, FeatureSettings
from src.services.temporal_features import event_age_days

THREAD_EVENTS = ("open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment")
REVIEW_EVENTS = ("pr_review", "pr_review_comment")


def count_by_user(
    *,
    values: pd.Series,
    user_ids: pd.Index,
) -> pd.Series:
    return values.value_counts().reindex(user_ids).fillna(0.0).astype("float32")


def is_open_at(
    *,
    closed_at: pd.Series,
    merged_at: pd.Series | None,
    time: pd.Timestamp,
) -> pd.Series:
    closed = closed_at.notna() & (closed_at < time)
    if merged_at is not None:
        closed = closed | (merged_at.notna() & (merged_at < time))
    return ~closed


class RepositoryContextBlock:
    def __init__(self, *, settings: FeatureSettings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "repository_context"

    def build(self, *, context: FeatureContext) -> pd.DataFrame:
        user_ids = context.user_ids
        known = context.known_events.dropna(subset=["repo_id"])
        age = event_age_days(
            events=known,
            feature_time=context.feature_time,
        )
        lookback = known[age < self._settings.repo_lookback_days]
        mine = lookback[lookback["actor_id"].isin(user_ids)]
        user_repo = mine.groupby(["actor_id", "repo_id"]).size().rename("my_events")

        frame = pd.DataFrame(index=user_ids)
        frame["repos_28d"] = user_repo.groupby(level="actor_id").size().reindex(user_ids).fillna(0.0)
        totals = user_repo.groupby(level="actor_id").sum()
        frame["top_repo_share"] = (user_repo.groupby(level="actor_id").max() / totals).reindex(user_ids).fillna(0.0)

        org_repos = context.database.repositories.set_index("repo_id")["org_id"].notna()
        in_org = user_repo.index.get_level_values("repo_id").map(org_repos).fillna(False).astype(bool)
        org_events = user_repo[np.asarray(in_org)].groupby(level="actor_id").sum()
        frame["org_repo_share"] = (org_events / totals).reindex(user_ids).fillna(0.0)
        heat = self._repository_heat(
            known=known,
            age=age,
            user_repo=user_repo,
            user_ids=user_ids,
        )
        return frame.join(heat).astype("float32")

    def _repository_heat(
        self,
        *,
        known: pd.DataFrame,
        age: pd.Series,
        user_repo: pd.Series,
        user_ids: pd.Index,
    ) -> pd.DataFrame:
        recent = known[age < self._settings.repo_heat_days]
        repo_events = recent.groupby(["repo_id", "event"], observed=True).size().unstack("event", fill_value=0)
        repo_events.columns = repo_events.columns.astype(str)
        repo_events = repo_events.reindex(columns=list(EVENTS), fill_value=0)
        repo_actors = recent.groupby("repo_id")["actor_id"].nunique().rename("actors")

        own = recent[recent["actor_id"].isin(user_ids)]
        own_events = own.groupby(["actor_id", "repo_id", "event"], observed=True).size().unstack("event", fill_value=0)
        own_events.columns = own_events.columns.astype(str)
        own_events = own_events.reindex(columns=list(EVENTS), fill_value=0)

        pairs = user_repo.reset_index()[["actor_id", "repo_id"]]
        merged = pairs.merge(
            repo_events.add_prefix("repo_").reset_index(),
            on="repo_id",
            how="left",
        ).merge(
            own_events.add_prefix("own_").reset_index(),
            on=["actor_id", "repo_id"],
            how="left",
        ).merge(
            repo_actors.reset_index(),
            on="repo_id",
            how="left",
        ).fillna(0.0)

        heat = pd.DataFrame(index=user_ids)
        for event in EVENTS:
            others = (merged[f"repo_{event}"] - merged[f"own_{event}"]).clip(lower=0.0)
            heat[f"repo_others_{event}_7d"] = others.groupby(merged["actor_id"]).sum().reindex(user_ids).fillna(0.0)
        own_any = merged[[f"own_{event}" for event in EVENTS]].sum(axis=1).gt(0)
        other_actors = (merged["actors"] - own_any.astype(float)).clip(lower=0.0)
        heat["repo_other_actors_7d"] = other_actors.groupby(merged["actor_id"]).sum().reindex(user_ids).fillna(0.0)
        repo_total = merged[[f"repo_{event}" for event in EVENTS]].sum(axis=1)
        heat["repo_max_events_7d"] = repo_total.groupby(merged["actor_id"]).max().reindex(user_ids).fillna(0.0)
        return heat


class CollaborationBlock:
    def __init__(self, *, settings: FeatureSettings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "collaboration"

    def build(self, *, context: FeatureContext) -> pd.DataFrame:
        user_ids = context.user_ids
        time = context.feature_time
        database = context.database
        frame = pd.DataFrame(index=user_ids)

        pull_requests = database.pull_requests[database.pull_requests["first_seen_at"] < time]
        pr_open = is_open_at(
            closed_at=pull_requests["closed_at"],
            merged_at=pull_requests["merged_at"],
            time=time,
        )
        open_since = time - pd.Timedelta(days=self._settings.open_item_days)
        recent_created = pull_requests["created_at"].between(open_since, time, inclusive="left")
        frame["open_prs_authored"] = count_by_user(
            values=pull_requests.loc[pr_open & recent_created, "author_id"],
            user_ids=user_ids,
        )
        merged_since = time - pd.Timedelta(days=self._settings.collaboration_days)
        recently_merged = pull_requests["merged_at"].notna() & pull_requests["merged_at"].between(merged_since, time, inclusive="left")
        frame["merged_prs_28d"] = count_by_user(
            values=pull_requests.loc[recently_merged, "author_id"],
            user_ids=user_ids,
        )

        received_since = time - pd.Timedelta(days=self._settings.received_days)
        pr_author = pull_requests.set_index("pr_id")["author_id"]
        reviews = database.pr_reviews[
            (database.pr_reviews["first_seen_at"] < time)
            & database.pr_reviews["submitted_at"].between(received_since, time, inclusive="left")
        ]
        frame["reviews_received_14d"] = self._received(
            target_author=reviews["pr_id"].map(pr_author),
            actor=reviews["reviewer_id"],
            user_ids=user_ids,
        )
        review_comments = database.pr_review_comments[
            database.pr_review_comments["created_at"].between(received_since, time, inclusive="left")
        ]
        frame["review_comments_received_14d"] = self._received(
            target_author=review_comments["pr_id"].map(pr_author),
            actor=review_comments["author_id"],
            user_ids=user_ids,
        )
        issues = database.issues[database.issues["first_seen_at"] < time]
        issue_author = issues.set_index("issue_id")["author_id"]
        comments = database.issue_comments[database.issue_comments["created_at"].between(received_since, time, inclusive="left")]
        frame["comments_received_14d"] = self._received(
            target_author=comments["issue_id"].map(issue_author),
            actor=comments["author_id"],
            user_ids=user_ids,
        )

        issue_open = is_open_at(
            closed_at=issues["closed_at"],
            merged_at=None,
            time=time,
        )
        plain = ~issues["is_pull_request"].fillna(False).astype(bool)
        issue_recent = issues["created_at"].between(open_since, time, inclusive="left")
        frame["open_issues_authored"] = count_by_user(
            values=issues.loc[plain & issue_open & issue_recent, "author_id"],
            user_ids=user_ids,
        )
        frame["open_issues_assigned"] = count_by_user(
            values=issues.loc[issue_open & issues["assignee_id"].notna(), "assignee_id"],
            user_ids=user_ids,
        )
        frame["open_prs_reviewed"] = self._open_reviewed_prs(
            context=context,
            pull_requests=pull_requests,
            pr_open=pr_open,
        )
        return frame.join(self._thread_degree(context=context)).astype("float32")

    @staticmethod
    def _received(
        *,
        target_author: pd.Series,
        actor: pd.Series,
        user_ids: pd.Index,
    ) -> pd.Series:
        from_others = target_author.notna() & (target_author != actor)
        return count_by_user(
            values=target_author[from_others],
            user_ids=user_ids,
        )

    def _open_reviewed_prs(
        self,
        *,
        context: FeatureContext,
        pull_requests: pd.DataFrame,
        pr_open: pd.Series,
    ) -> pd.Series:
        known = context.known_events
        age = event_age_days(
            events=known,
            feature_time=context.feature_time,
        )
        reviewed = known[
            known["event"].isin(REVIEW_EVENTS)
            & (age < self._settings.collaboration_days)
            & known["actor_id"].isin(context.user_ids)
            & known["pr_id"].notna()
        ]
        pairs = reviewed[["actor_id", "pr_id"]].drop_duplicates()
        open_authors = pull_requests.loc[pr_open, ["pr_id", "author_id"]]
        pairs = pairs.merge(
            open_authors,
            on="pr_id",
            how="inner",
        )
        pairs = pairs[pairs["author_id"] != pairs["actor_id"]]
        return count_by_user(
            values=pairs["actor_id"],
            user_ids=context.user_ids,
        )

    def _thread_degree(self, *, context: FeatureContext) -> pd.DataFrame:
        known = context.known_events
        age = event_age_days(
            events=known,
            feature_time=context.feature_time,
        )
        threads = known[known["event"].isin(THREAD_EVENTS) & (age < self._settings.collaboration_days)].copy()
        threads["thread"] = np.where(
            threads["pr_id"].notna(),
            threads["pr_id"],
            -1.0 - threads["issue_id"],
        )
        threads = threads.dropna(subset=["thread"])
        thread_size = threads.groupby("thread")["actor_id"].nunique().rename("thread_actors")
        mine = threads[threads["actor_id"].isin(context.user_ids)][["actor_id", "thread"]].drop_duplicates()
        mine = mine.merge(
            thread_size.reset_index(),
            on="thread",
            how="left",
        )
        degree = pd.DataFrame(index=context.user_ids)
        degree["threads_28d"] = mine.groupby("actor_id").size().reindex(context.user_ids).fillna(0.0)
        degree["collaborators_28d"] = (mine["thread_actors"] - 1).groupby(mine["actor_id"]).sum().reindex(context.user_ids).fillna(0.0)
        return degree
