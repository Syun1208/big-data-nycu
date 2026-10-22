from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.classes.settings import EVENTS, STGNNSettings
from src.data.loaders.event_log import EventLog

SECONDS_PER_DAY = 86400.0


class ActivitySeriesBuilder:
    def __init__(
        self,
        *,
        event_log: EventLog,
        settings: STGNNSettings,
    ) -> None:
        self._event_log = event_log
        self._settings = settings

    def build(
        self,
        *,
        user_ids: np.ndarray,
        feature_time: pd.Timestamp,
    ) -> np.ndarray:
        steps = self._settings.num_steps
        user_ids = np.asarray(user_ids, dtype="int64")
        known = self._event_log.known_before(time=feature_time)
        start = feature_time - pd.Timedelta(days=steps * self._settings.step_days)
        known = known[(known["event_time"] >= start) & known["actor_id"].isin(user_ids)]
        age_days = (feature_time - known["event_time"]).dt.total_seconds().to_numpy() / SECONDS_PER_DAY
        steps_back = np.minimum((age_days // self._settings.step_days).astype("int64"), steps - 1)
        counts = np.zeros((len(user_ids), steps, len(EVENTS)), dtype="float32")
        np.add.at(
            counts,
            (
                pd.Index(user_ids).get_indexer(known["actor_id"].to_numpy()),
                steps - 1 - steps_back,
                known["event"].cat.codes.to_numpy(dtype="int64"),
            ),
            1.0,
        )
        return np.log1p(counts)
