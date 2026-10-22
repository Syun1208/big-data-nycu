from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.classes.settings import STHNSettings
from src.data.loaders.event_log import EventLog

PAD_KIND = -1


@dataclass(frozen=True, slots=True)
class InteractionSequences:
    kinds: np.ndarray
    age_seconds: np.ndarray

    @property
    def size(self) -> int:
        return self.kinds.shape[0]

    @property
    def lengths(self) -> np.ndarray:
        return (self.kinds != PAD_KIND).sum(axis=1)


class InteractionSequenceBuilder:
    def __init__(
        self,
        *,
        event_log: EventLog,
        settings: STHNSettings,
    ) -> None:
        self._event_log = event_log
        self._settings = settings

    def build(
        self,
        *,
        user_ids: np.ndarray,
        feature_time: pd.Timestamp,
    ) -> InteractionSequences:
        max_edges = self._settings.max_edges
        user_ids = np.asarray(user_ids, dtype="int64")
        known = self._event_log.known_before(time=feature_time)
        start = feature_time - pd.Timedelta(days=self._settings.lookback_days)
        known = known[(known["event_time"] >= start) & known["actor_id"].isin(user_ids)]
        known = known.sort_values("event_time", ascending=False, kind="stable")
        rank = known.groupby("actor_id", sort=False).cumcount().to_numpy()
        recent = known[rank < max_edges]
        rank = rank[rank < max_edges]
        rows = pd.Index(user_ids).get_indexer(recent["actor_id"].to_numpy())
        kinds = np.full((len(user_ids), max_edges), PAD_KIND, dtype="int8")
        age_seconds = np.zeros((len(user_ids), max_edges), dtype="float32")
        kinds[rows, rank] = recent["event"].cat.codes.to_numpy(dtype="int8")
        age_seconds[rows, rank] = (feature_time - recent["event_time"]).dt.total_seconds().to_numpy(dtype="float32")
        return InteractionSequences(
            kinds=kinds,
            age_seconds=age_seconds,
        )
