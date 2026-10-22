from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders.relational_db import RelationalDatabase  # noqa: E402
from src.utils.env import load_env  # noqa: E402

load_env(start=ROOT)

T0 = pd.Timestamp("2026-09-01 00:00:00")


def at(*, days: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(days=days)


def build_database(*, extra_push_days: tuple[float, ...] = ()) -> RelationalDatabase:
    users = pd.DataFrame(
        {
            "user_id": [0, 1, 2],
            "github_user_id": [100, 101, 102],
            "login": ["alice", "bob", "carol"],
            "is_cohort": [True, True, False],
            "first_seen_at": [at(days=0), at(days=1), at(days=2)],
        }
    )
    repositories = pd.DataFrame(
        {
            "repo_id": [0, 1],
            "github_repo_id": [500, 501],
            "name": ["org/app", "bob/tool"],
            "org_id": pd.array([9, None], dtype="Int64"),
            "first_seen_at": [at(days=0), at(days=1)],
        }
    )
    pull_requests = pd.DataFrame(
        {
            "pr_id": [0, 1],
            "github_pr_id": [700, 701],
            "repo_id": [0, 0],
            "number": [1, 2],
            "author_id": pd.array([0, 2], dtype="Int64"),
            "created_at": [at(days=3), at(days=4)],
            "merged_at": [at(days=6), pd.NaT],
            "merged_by_id": pd.array([1, None], dtype="Int64"),
            "closed_at": [at(days=6), pd.NaT],
            "first_seen_at": [at(days=3), at(days=4)],
        }
    )
    issues = pd.DataFrame(
        {
            "issue_id": [0, 1],
            "github_issue_id": [800, 801],
            "repo_id": [0, 0],
            "number": [1, 3],
            "author_id": pd.array([0, 1], dtype="Int64"),
            "is_pull_request": [True, False],
            "pull_request_id": pd.array([0, None], dtype="Int64"),
            "assignee_id": pd.array([None, 0], dtype="Int64"),
            "created_at": [at(days=3), at(days=5)],
            "closed_at": [pd.NaT, pd.NaT],
            "first_seen_at": [at(days=3), at(days=5)],
        }
    )
    issue_comments = pd.DataFrame(
        {
            "comment_id": [0, 1],
            "github_comment_id": [900, 901],
            "issue_id": [0, 1],
            "repo_id": [0, 0],
            "author_id": pd.array([1, 0], dtype="Int64"),
            "created_at": [at(days=5), at(days=8)],
        }
    )
    pr_reviews = pd.DataFrame(
        {
            "review_id": [0],
            "github_review_id": [1000],
            "pr_id": [1],
            "repo_id": [0],
            "reviewer_id": pd.array([0], dtype="Int64"),
            "state": ["approved"],
            "submitted_at": [at(days=5)],
            "first_seen_at": [at(days=5)],
        }
    )
    pr_review_comments = pd.DataFrame(
        {
            "comment_id": [0],
            "github_comment_id": [1100],
            "pr_id": [0],
            "repo_id": [0],
            "author_id": pd.array([1], dtype="Int64"),
            "pull_request_review_id": pd.array([None], dtype="Int64"),
            "in_reply_to_id": pd.array([None], dtype="Int64"),
            "created_at": [at(days=5.5)],
        }
    )
    push_days = (1.0, 2.0, 4.0, 6.0, 9.0, *extra_push_days)
    pushes = pd.DataFrame(
        {
            "push_id": range(len(push_days) + 2),
            "github_push_id": range(2000, 2002 + len(push_days)),
            "user_id": pd.array([0] * len(push_days) + [1, 1], dtype="Int64"),
            "repo_id": pd.array([0] * len(push_days) + [1, 0], dtype="Int64"),
            "ref_type": ["branch"] * (len(push_days) + 2),
            "is_new_ref": [True] + [False] * (len(push_days) + 1),
            "created_at": [at(days=day) for day in push_days] + [at(days=2), at(days=7)],
        }
    )
    return RelationalDatabase(
        users=users,
        repositories=repositories,
        pull_requests=pull_requests,
        issues=issues,
        issue_comments=issue_comments,
        pr_reviews=pr_reviews,
        pr_review_comments=pr_review_comments,
        pushes=pushes,
    )


@pytest.fixture()
def database() -> RelationalDatabase:
    return build_database()
