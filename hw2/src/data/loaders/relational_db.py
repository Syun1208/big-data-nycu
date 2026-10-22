from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.progress import LOAD, create_progress

logger = logging.getLogger(__name__)

TABLE_COLUMNS = {
    "Users": ("user_id", "github_user_id", "login", "is_cohort", "first_seen_at"),
    "Repositories": ("repo_id", "github_repo_id", "name", "org_id", "first_seen_at"),
    "PullRequests": (
        "pr_id", "github_pr_id", "repo_id", "number", "author_id", "created_at",
        "merged_at", "merged_by_id", "closed_at", "first_seen_at",
    ),
    "Issues": (
        "issue_id", "github_issue_id", "repo_id", "number", "author_id", "is_pull_request",
        "pull_request_id", "assignee_id", "created_at", "closed_at", "first_seen_at",
    ),
    "IssueComments": ("comment_id", "github_comment_id", "issue_id", "repo_id", "author_id", "created_at"),
    "PRReviews": ("review_id", "github_review_id", "pr_id", "repo_id", "reviewer_id", "state", "submitted_at", "first_seen_at"),
    "PRReviewComments": (
        "comment_id", "github_comment_id", "pr_id", "repo_id", "author_id",
        "pull_request_review_id", "in_reply_to_id", "created_at",
    ),
    "Pushes": ("push_id", "github_push_id", "user_id", "repo_id", "ref_type", "is_new_ref", "created_at"),
}


@dataclass(frozen=True, slots=True)
class RelationalDatabase:
    users: pd.DataFrame
    repositories: pd.DataFrame
    pull_requests: pd.DataFrame
    issues: pd.DataFrame
    issue_comments: pd.DataFrame
    pr_reviews: pd.DataFrame
    pr_review_comments: pd.DataFrame
    pushes: pd.DataFrame

    @property
    def cohort_user_ids(self) -> pd.Index:
        return pd.Index(self.users.loc[self.users["is_cohort"], "user_id"])

    def github_to_user_ids(self) -> pd.Series:
        return self.users.set_index("github_user_id")["user_id"]

    def user_to_github_ids(self) -> pd.Series:
        return self.users.set_index("user_id")["github_user_id"]

    def latest_row_time(self) -> pd.Timestamp:
        return max(
            self.pushes["created_at"].max(),
            self.issue_comments["created_at"].max(),
            self.pr_review_comments["created_at"].max(),
            self.pull_requests["first_seen_at"].max(),
            self.issues["first_seen_at"].max(),
            self.pr_reviews["first_seen_at"].max(),
        )


class ParquetDatabaseLoader:
    def __init__(self, *, db_dir: Path) -> None:
        self._db_dir = db_dir

    def load(self) -> RelationalDatabase:
        if not self._db_dir.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {self._db_dir}. Run `bash scripts/download_data.sh` first.")
        tables = {}
        with create_progress(
            stage=LOAD,
            total=len(TABLE_COLUMNS),
            unit="table",
        ) as bar:
            for name, columns in TABLE_COLUMNS.items():
                tables[name] = pd.read_parquet(
                    self._db_dir / f"{name}.parquet",
                    columns=list(columns),
                )
                logger.debug("Loaded %s: %d rows", name, len(tables[name]))
                bar.set_postfix_str(name)
                bar.update(1)
        database = RelationalDatabase(
            users=tables["Users"],
            repositories=tables["Repositories"],
            pull_requests=tables["PullRequests"],
            issues=tables["Issues"],
            issue_comments=tables["IssueComments"],
            pr_reviews=tables["PRReviews"],
            pr_review_comments=tables["PRReviewComments"],
            pushes=tables["Pushes"],
        )
        logger.info(
            "Loaded %d tables from %s (%d users, %d cohort, data until %s)",
            len(tables),
            self._db_dir,
            len(database.users),
            len(database.cohort_user_ids),
            database.latest_row_time(),
        )
        return database
