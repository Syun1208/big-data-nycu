from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.data.classes.samples import PREDICT_SPLIT, Cutoff, CutoffDataset, CutoffSample, FeatureContext
from src.data.classes.settings import EVENTS
from src.data.loaders.event_log import EventLog
from src.data.loaders.relational_db import RelationalDatabase
from src.interface.feature_block import FeatureBlock
from src.services.sampling import LabelBuilder, PopulationSelector
from src.utils.progress import FEATURES, create_progress

logger = logging.getLogger(__name__)

GITHUB_ID_COLUMN = "github_user_id"
CUTOFF_COLUMN = "cutoff"
SPLIT_COLUMN = "split"
META_COLUMNS = (GITHUB_ID_COLUMN, CUTOFF_COLUMN, SPLIT_COLUMN)
LABEL_PREFIX = "label_"


class FeatureAssembler:
    def __init__(self, *, blocks: Sequence[FeatureBlock]) -> None:
        self._blocks = tuple(blocks)

    @property
    def block_names(self) -> list[str]:
        return [block.name for block in self._blocks]

    def build(self, *, context: FeatureContext) -> pd.DataFrame:
        parts = []
        for block in self._blocks:
            part = block.build(context=context)
            logger.debug("Block %s: %d columns", block.name, part.shape[1])
            parts.append(part)
        features = pd.concat(parts, axis=1)
        duplicated = features.columns[features.columns.duplicated()]
        if len(duplicated):
            raise ValueError(f"Duplicated feature columns: {list(duplicated)}")
        return features.astype("float32")


class DatasetBuilder:
    def __init__(
        self,
        *,
        database: RelationalDatabase,
        event_log: EventLog,
        selector: PopulationSelector,
        label_builder: LabelBuilder,
        assembler: FeatureAssembler,
        cache_dir: Path | None,
        cache_key: str,
    ) -> None:
        self._database = database
        self._event_log = event_log
        self._selector = selector
        self._label_builder = label_builder
        self._assembler = assembler
        self._cache_dir = cache_dir
        self._cache_key = cache_key

    def build_many(self, *, cutoffs: Sequence[Cutoff]) -> list[CutoffSample]:
        samples = []
        with create_progress(
            stage=FEATURES,
            total=len(cutoffs),
            unit="week",
        ) as bar:
            for cutoff in cutoffs:
                bar.set_postfix_str(f"{cutoff.split} {cutoff.label}")
                samples.append(
                    CutoffSample(
                        cutoff=cutoff,
                        dataset=self.build(cutoff=cutoff),
                    )
                )
                bar.update(1)
        return samples

    def build(
        self,
        *,
        cutoff: Cutoff,
        user_ids: pd.Index | None = None,
    ) -> CutoffDataset:
        cached = self._read_cache(cutoff=cutoff) if user_ids is None else None
        if cached is not None:
            return cached
        if user_ids is None:
            user_ids = self._selector.select(
                event_log=self._event_log,
                cohort_user_ids=self._database.cohort_user_ids,
                cutoff=cutoff,
            )
        context = FeatureContext(
            user_ids=user_ids,
            feature_time=cutoff.feature_time,
            known_events=self._event_log.known_before(time=cutoff.feature_time),
            database=self._database,
        )
        features = self._assembler.build(context=context)
        labels = None
        if cutoff.split != PREDICT_SPLIT:
            labels = self._label_builder.build(
                event_log=self._event_log,
                user_ids=user_ids,
                cutoff=cutoff,
            )
        meta = pd.DataFrame(
            {
                GITHUB_ID_COLUMN: user_ids.map(self._database.user_to_github_ids()).astype("int64"),
                CUTOFF_COLUMN: cutoff.label,
                SPLIT_COLUMN: cutoff.split,
            },
            index=user_ids,
        )
        dataset = CutoffDataset(
            features=features.reset_index(drop=True),
            labels=None if labels is None else labels.reset_index(drop=True),
            meta=meta.reset_index(names="user_id"),
        )
        logger.info(
            "Cutoff %s (%s): %d users, %d features%s",
            cutoff.label,
            cutoff.split,
            len(user_ids),
            features.shape[1],
            "" if labels is None else ", positive rate " + ", ".join(f"{event}={labels[event].mean():.2f}" for event in EVENTS),
        )
        if cutoff.split != PREDICT_SPLIT:
            self._write_cache(
                cutoff=cutoff,
                dataset=dataset,
            )
        return dataset

    def _cache_path(self, *, cutoff: Cutoff) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{cutoff.time:%Y%m%d_%H%M}_{self._cache_key}.parquet"

    def _read_cache(self, *, cutoff: Cutoff) -> CutoffDataset | None:
        path = self._cache_path(cutoff=cutoff)
        if path is None or not path.is_file():
            return None
        frame = pd.read_parquet(path)
        label_columns = [f"{LABEL_PREFIX}{event}" for event in EVENTS]
        meta_columns = ["user_id", *META_COLUMNS]
        feature_columns = [column for column in frame.columns if column not in label_columns and column not in meta_columns]
        labels = frame[label_columns].rename(columns=lambda column: column.removeprefix(LABEL_PREFIX))
        meta = frame[meta_columns].copy()
        meta[SPLIT_COLUMN] = cutoff.split
        logger.debug("Loaded cached cutoff %s from %s", cutoff.label, path)
        return CutoffDataset(
            features=frame[feature_columns],
            labels=labels.astype("int8"),
            meta=meta,
        )

    def _write_cache(
        self,
        *,
        cutoff: Cutoff,
        dataset: CutoffDataset,
    ) -> None:
        path = self._cache_path(cutoff=cutoff)
        if path is None or dataset.labels is None:
            return
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        frame = pd.concat(
            [
                dataset.meta,
                dataset.features,
                dataset.labels.add_prefix(LABEL_PREFIX),
            ],
            axis=1,
        )
        partial = path.with_suffix(f".{os.getpid()}.partial")
        frame.to_parquet(
            partial,
            index=False,
        )
        partial.replace(path)


def build_cache_key(
    *,
    settings: dict,
    data_end: pd.Timestamp,
    block_names: Sequence[str],
) -> str:
    payload = json.dumps(
        {
            "settings": settings,
            "data_end": str(data_end),
            "blocks": list(block_names),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
