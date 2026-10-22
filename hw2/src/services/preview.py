from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from src.data.classes.samples import CutoffSample
from src.data.classes.settings import EVENTS

logger = logging.getLogger(__name__)

PREVIEW_ROWS = 5
PREVIEW_SEED = 0
META_PREVIEW_COLUMNS = ("user_id", "github_user_id", "cutoff", "split")
FEATURE_PREVIEW_COLUMNS = (
    "count_push_7d",
    "count_any_28d",
    "days_since_any",
    "week_rate_push",
    "streak_any",
    "repos_28d",
    "threads_28d",
    "open_prs_authored",
)


def preview_samples(
    *,
    title: str,
    samples: Sequence[CutoffSample],
    rows: int = PREVIEW_ROWS,
) -> pd.DataFrame:
    parts = [pd.concat([sample.dataset.meta for sample in samples], ignore_index=True)[list(META_PREVIEW_COLUMNS)]]
    features = pd.concat([sample.dataset.features for sample in samples], ignore_index=True)
    parts.append(features[[column for column in FEATURE_PREVIEW_COLUMNS if column in features.columns]])
    labels = [sample.dataset.labels for sample in samples]
    if all(label is not None for label in labels):
        parts.append(pd.concat(labels, ignore_index=True)[list(EVENTS)].add_prefix("y_"))
    table = pd.concat(parts, axis=1)
    preview = table.sample(n=min(rows, len(table)), random_state=PREVIEW_SEED)
    logger.info(
        "🔎 %s: %d rows x %d features, %d sample rows:\n%s",
        title,
        len(table),
        features.shape[1],
        len(preview),
        preview.to_string(index=False, float_format=lambda value: f"{value:.2f}"),
    )
    return preview
