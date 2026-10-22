from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.classes.settings import EVENTS
from src.data.loaders.relational_db import RelationalDatabase

logger = logging.getLogger(__name__)

EVENT_LOG_COLUMNS = ("actor_id", "event", "event_time", "row_time", "repo_id", "pr_id", "issue_id")


@dataclass(frozen=True, slots=True)
class EventLog:
    frame: pd.DataFrame

    def known_before(self, *, time: pd.Timestamp) -> pd.DataFrame:
        frame = self.frame
        return frame[(frame["row_time"] < time) & (frame["event_time"] < time)]

    def occurring_between(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        frame = self.frame
        return frame[(frame["event_time"] >= start) & (frame["event_time"] < end)]


def build_event_log(*, database: RelationalDatabase) -> EventLog:
    pull_requests = database.pull_requests
    issues = database.issues
    plain_issues = issues[~issues["is_pull_request"].fillna(False).astype(bool)]
    comments = database.issue_comments.merge(
        issues[["issue_id", "pull_request_id"]],
        on="issue_id",
        how="left",
    )
    reviews = database.pr_reviews
    review_comments = database.pr_review_comments
    pushes = database.pushes

    parts = [
        _event_frame(
            event="open_pr",
            actor=pull_requests["author_id"],
            event_time=pull_requests["created_at"],
            row_time=pull_requests["first_seen_at"],
            repo=pull_requests["repo_id"],
            pr=pull_requests["pr_id"],
            issue=None,
        ),
        _event_frame(
            event="open_issue",
            actor=plain_issues["author_id"],
            event_time=plain_issues["created_at"],
            row_time=plain_issues["first_seen_at"],
            repo=plain_issues["repo_id"],
            pr=None,
            issue=plain_issues["issue_id"],
        ),
        _event_frame(
            event="issue_comment",
            actor=comments["author_id"],
            event_time=comments["created_at"],
            row_time=comments["created_at"],
            repo=comments["repo_id"],
            pr=comments["pull_request_id"],
            issue=comments["issue_id"],
        ),
        _event_frame(
            event="pr_review",
            actor=reviews["reviewer_id"],
            event_time=reviews["submitted_at"],
            row_time=reviews["first_seen_at"],
            repo=reviews["repo_id"],
            pr=reviews["pr_id"],
            issue=None,
        ),
        _event_frame(
            event="pr_review_comment",
            actor=review_comments["author_id"],
            event_time=review_comments["created_at"],
            row_time=review_comments["created_at"],
            repo=review_comments["repo_id"],
            pr=review_comments["pr_id"],
            issue=None,
        ),
        _event_frame(
            event="push",
            actor=pushes["user_id"],
            event_time=pushes["created_at"],
            row_time=pushes["created_at"],
            repo=pushes["repo_id"],
            pr=None,
            issue=None,
        ),
    ]
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.dropna(subset=["actor_id", "event_time", "row_time"])
    frame["actor_id"] = frame["actor_id"].astype(np.int64)
    frame["event"] = pd.Categorical(frame["event"], categories=list(EVENTS))
    frame = frame.sort_values("event_time", kind="stable").reset_index(drop=True)
    logger.info("Event log: %d events from %d actors", len(frame), frame["actor_id"].nunique())
    return EventLog(frame=frame.loc[:, list(EVENT_LOG_COLUMNS)])


def _event_frame(
    *,
    event: str,
    actor: pd.Series,
    event_time: pd.Series,
    row_time: pd.Series,
    repo: pd.Series,
    pr: pd.Series | None,
    issue: pd.Series | None,
) -> pd.DataFrame:
    size = len(actor)
    return pd.DataFrame(
        {
            "actor_id": _as_float(series=actor),
            "event": event,
            "event_time": event_time.to_numpy(),
            "row_time": row_time.to_numpy(),
            "repo_id": _as_float(series=repo),
            "pr_id": _as_float(series=pr) if pr is not None else np.full(size, np.nan),
            "issue_id": _as_float(series=issue) if issue is not None else np.full(size, np.nan),
        }
    )


def _as_float(*, series: pd.Series) -> np.ndarray:
    return series.astype("Float64").to_numpy(
        dtype="float64",
        na_value=np.nan,
    )
