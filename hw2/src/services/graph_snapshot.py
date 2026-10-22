from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.classes.settings import EVENTS, GnnSettings
from src.data.loaders.event_log import EventLog
from src.data.loaders.relational_db import RelationalDatabase
from src.services.temporal_features import SECONDS_PER_DAY

logger = logging.getLogger(__name__)

NODE_TYPES = ("user", "repo", "pr", "issue")
AUTHOR_KIND = len(EVENTS)
ASSIGNEE_KIND = len(EVENTS) + 1
CONTAINS_KIND = len(EVENTS) + 2
NUM_EDGE_KINDS = len(EVENTS) + 3
AGE_SCALE = 6.0
NODE_FEATURE_DIMS = {"user": 3, "repo": 3, "pr": 5, "issue": 5}


@dataclass(frozen=True, slots=True)
class EdgeSet:
    source_type: str
    target_type: str
    source: np.ndarray
    target: np.ndarray
    kind: np.ndarray
    age_days: np.ndarray

    @property
    def size(self) -> int:
        return len(self.source)


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    feature_time: pd.Timestamp
    node_features: dict[str, np.ndarray]
    relations: dict[str, EdgeSet]

    @property
    def num_edges(self) -> int:
        return sum(edges.size for edges in self.relations.values())


def days_before(
    *,
    times: pd.Series,
    feature_time: pd.Timestamp,
) -> np.ndarray:
    return ((feature_time - times).dt.total_seconds() / SECONDS_PER_DAY).to_numpy(dtype="float32", na_value=np.nan)


def scaled_age(*, age: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.nan_to_num(age, nan=0.0), 0.0, None)) / AGE_SCALE


def known_mask(
    *,
    times: pd.Series,
    feature_time: pd.Timestamp,
) -> np.ndarray:
    return (times.notna() & (times < feature_time)).to_numpy()


def edge_set(
    *,
    source_type: str,
    target_type: str,
    source: pd.Series | np.ndarray,
    target: pd.Series | np.ndarray,
    kind: np.ndarray,
    age_days: np.ndarray,
) -> EdgeSet:
    source = np.asarray(source, dtype="float64")
    target = np.asarray(target, dtype="float64")
    valid = ~np.isnan(source) & ~np.isnan(target)
    return EdgeSet(
        source_type=source_type,
        target_type=target_type,
        source=source[valid].astype("int64"),
        target=target[valid].astype("int64"),
        kind=np.asarray(kind)[valid].astype("int64"),
        age_days=np.nan_to_num(np.asarray(age_days, dtype="float32")[valid], nan=0.0),
    )


def concat_edges(*, parts: list[EdgeSet]) -> EdgeSet:
    return EdgeSet(
        source_type=parts[0].source_type,
        target_type=parts[0].target_type,
        source=np.concatenate([part.source for part in parts]),
        target=np.concatenate([part.target for part in parts]),
        kind=np.concatenate([part.kind for part in parts]),
        age_days=np.concatenate([part.age_days for part in parts]),
    )


class GraphSnapshotBuilder:
    def __init__(
        self,
        *,
        database: RelationalDatabase,
        event_log: EventLog,
        settings: GnnSettings,
    ) -> None:
        self._database = database
        self._event_log = event_log
        self._settings = settings
        self._num_nodes = {
            "user": int(database.users["user_id"].max()) + 1,
            "repo": int(database.repositories["repo_id"].max()) + 1,
            "pr": int(database.pull_requests["pr_id"].max()) + 1,
            "issue": int(database.issues["issue_id"].max()) + 1,
        }
        self._event_codes = {event: code for code, event in enumerate(EVENTS)}

    @property
    def num_nodes(self) -> dict[str, int]:
        return dict(self._num_nodes)

    def build(self, *, feature_time: pd.Timestamp) -> GraphSnapshot:
        known = self._event_log.known_before(time=feature_time)
        age = days_before(times=known["event_time"], feature_time=feature_time)
        recent = known[age < self._settings.lookback_days]
        recent_age = age[age < self._settings.lookback_days]
        kind = recent["event"].astype(str).map(self._event_codes).to_numpy(dtype="int64")

        on_pr = recent["pr_id"].notna().to_numpy()
        on_issue = recent["issue_id"].notna().to_numpy() & ~on_pr
        relations = {
            "user__repo": edge_set(
                source_type="user",
                target_type="repo",
                source=recent["actor_id"],
                target=recent["repo_id"],
                kind=kind,
                age_days=recent_age,
            ),
            "user__pr": concat_edges(
                parts=[
                    edge_set(
                        source_type="user",
                        target_type="pr",
                        source=recent["actor_id"].to_numpy()[on_pr],
                        target=recent["pr_id"].to_numpy()[on_pr],
                        kind=kind[on_pr],
                        age_days=recent_age[on_pr],
                    ),
                    *self._pull_request_authors(feature_time=feature_time),
                ]
            ),
            "user__issue": concat_edges(
                parts=[
                    edge_set(
                        source_type="user",
                        target_type="issue",
                        source=recent["actor_id"].to_numpy()[on_issue],
                        target=recent["issue_id"].to_numpy()[on_issue],
                        kind=kind[on_issue],
                        age_days=recent_age[on_issue],
                    ),
                    *self._issue_people(feature_time=feature_time),
                ]
            ),
            **self._containment(feature_time=feature_time),
        }
        snapshot = GraphSnapshot(
            feature_time=feature_time,
            node_features=self._node_features(feature_time=feature_time),
            relations=relations,
        )
        logger.debug(
            "Graph snapshot at %s: %s",
            feature_time,
            ", ".join(f"{name}={edges.size}" for name, edges in relations.items()),
        )
        return snapshot

    def _active_items(
        self,
        *,
        frame: pd.DataFrame,
        feature_time: pd.Timestamp,
        merged: bool,
    ) -> pd.DataFrame:
        seen = known_mask(times=frame["first_seen_at"], feature_time=feature_time)
        closed = known_mask(times=frame["closed_at"], feature_time=feature_time)
        if merged:
            closed = closed | known_mask(times=frame["merged_at"], feature_time=feature_time)
        created_age = days_before(times=frame["created_at"].fillna(frame["first_seen_at"]), feature_time=feature_time)
        recent = created_age < self._settings.lookback_days
        return frame[seen & (~closed | recent)]

    def _pull_request_authors(self, *, feature_time: pd.Timestamp) -> list[EdgeSet]:
        active = self._active_items(
            frame=self._database.pull_requests,
            feature_time=feature_time,
            merged=True,
        )
        age = days_before(times=active["created_at"].fillna(active["first_seen_at"]), feature_time=feature_time)
        return [
            edge_set(
                source_type="user",
                target_type="pr",
                source=active["author_id"].astype("Float64").to_numpy(dtype="float64", na_value=np.nan),
                target=active["pr_id"],
                kind=np.full(len(active), AUTHOR_KIND),
                age_days=age,
            )
        ]

    def _issue_people(self, *, feature_time: pd.Timestamp) -> list[EdgeSet]:
        active = self._active_items(
            frame=self._database.issues,
            feature_time=feature_time,
            merged=False,
        )
        age = days_before(times=active["created_at"].fillna(active["first_seen_at"]), feature_time=feature_time)
        return [
            edge_set(
                source_type="user",
                target_type="issue",
                source=active[column].astype("Float64").to_numpy(dtype="float64", na_value=np.nan),
                target=active["issue_id"],
                kind=np.full(len(active), kind),
                age_days=age,
            )
            for column, kind in (("author_id", AUTHOR_KIND), ("assignee_id", ASSIGNEE_KIND))
        ]

    def _containment(self, *, feature_time: pd.Timestamp) -> dict[str, EdgeSet]:
        relations = {}
        for name, frame, key, merged in (
            ("pr__repo", self._database.pull_requests, "pr_id", True),
            ("issue__repo", self._database.issues, "issue_id", False),
        ):
            active = self._active_items(
                frame=frame,
                feature_time=feature_time,
                merged=merged,
            )
            relations[name] = edge_set(
                source_type=name.split("__")[0],
                target_type="repo",
                source=active[key],
                target=active["repo_id"].astype("Float64").to_numpy(dtype="float64", na_value=np.nan),
                kind=np.full(len(active), CONTAINS_KIND),
                age_days=days_before(times=active["created_at"].fillna(active["first_seen_at"]), feature_time=feature_time),
            )
        issues = self._active_items(
            frame=self._database.issues,
            feature_time=feature_time,
            merged=False,
        )
        conversation = issues["pull_request_id"].notna()
        relations["issue__pr"] = edge_set(
            source_type="issue",
            target_type="pr",
            source=issues.loc[conversation, "issue_id"],
            target=issues.loc[conversation, "pull_request_id"].astype("Float64").to_numpy(dtype="float64", na_value=np.nan),
            kind=np.full(int(conversation.sum()), CONTAINS_KIND),
            age_days=np.zeros(int(conversation.sum()), dtype="float32"),
        )
        return relations

    def _node_features(self, *, feature_time: pd.Timestamp) -> dict[str, np.ndarray]:
        database = self._database
        features = {name: np.zeros((count, NODE_FEATURE_DIMS[name]), dtype="float32") for name, count in self._num_nodes.items()}

        users = database.users
        seen = known_mask(times=users["first_seen_at"], feature_time=feature_time)
        rows = users["user_id"].to_numpy()
        features["user"][rows, 0] = seen
        features["user"][rows, 1] = np.where(seen, users["is_cohort"].to_numpy(dtype="float32"), 0.0)
        features["user"][rows, 2] = np.where(seen, scaled_age(age=days_before(times=users["first_seen_at"], feature_time=feature_time)), 0.0)

        repositories = database.repositories
        seen = known_mask(times=repositories["first_seen_at"], feature_time=feature_time)
        rows = repositories["repo_id"].to_numpy()
        features["repo"][rows, 0] = seen
        features["repo"][rows, 1] = np.where(seen, repositories["org_id"].notna().to_numpy(dtype="float32"), 0.0)
        features["repo"][rows, 2] = np.where(seen, scaled_age(age=days_before(times=repositories["first_seen_at"], feature_time=feature_time)), 0.0)

        pull_requests = database.pull_requests
        seen = known_mask(times=pull_requests["first_seen_at"], feature_time=feature_time)
        merged = known_mask(times=pull_requests["merged_at"], feature_time=feature_time)
        closed = known_mask(times=pull_requests["closed_at"], feature_time=feature_time) | merged
        rows = pull_requests["pr_id"].to_numpy()
        features["pr"][rows, 0] = seen
        features["pr"][rows, 1] = np.where(seen, scaled_age(age=days_before(times=pull_requests["created_at"], feature_time=feature_time)), 0.0)
        features["pr"][rows, 2] = seen & ~closed
        features["pr"][rows, 3] = seen & merged
        features["pr"][rows, 4] = seen & pull_requests["created_at"].notna().to_numpy()

        issues = database.issues
        seen = known_mask(times=issues["first_seen_at"], feature_time=feature_time)
        closed = known_mask(times=issues["closed_at"], feature_time=feature_time)
        rows = issues["issue_id"].to_numpy()
        features["issue"][rows, 0] = seen
        features["issue"][rows, 1] = np.where(seen, scaled_age(age=days_before(times=issues["created_at"], feature_time=feature_time)), 0.0)
        features["issue"][rows, 2] = seen & ~closed
        features["issue"][rows, 3] = seen & issues["is_pull_request"].fillna(False).to_numpy(dtype=bool)
        features["issue"][rows, 4] = seen & issues["assignee_id"].notna().to_numpy()
        return features
