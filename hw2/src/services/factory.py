from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.classes.settings import DataSettings, ExperimentSettings, FeatureSettings
from src.data.loaders.event_log import EventLog, build_event_log
from src.data.loaders.relational_db import ParquetDatabaseLoader, RelationalDatabase
from src.interface.activity_model import ActivityModel
from src.interface.event_classifier import EventClassifier
from src.interface.feature_block import FeatureBlock
from src.services.behavior_features import BehaviorBlock
from src.services.config_loader import ConfigError, settings_to_dict
from src.services.dataset_builder import DatasetBuilder, FeatureAssembler, build_cache_key
from src.services.prediction import SubmissionPredictor
from src.services.relational_features import CollaborationBlock, RepositoryContextBlock
from src.services.sampling import LabelBuilder, PopulationSelector, SamplePlanner
from src.services.scoring import ThresholdTuner
from src.services.temporal_features import ActivityHistoryBlock, ProfileBlock
from src.services.training import ExperimentTrainer, RunPaths

XGBOOST_MODEL = "xgboost"
GNN_MODEL = "rdl_gnn"
TABPFN_MODEL = "tabpfn"
MODEL_TYPES = (XGBOOST_MODEL, GNN_MODEL, TABPFN_MODEL)
SPLIT_ONLY_SAMPLING_KEYS = ("anchor_cutoff", "train_start", "calibration_weeks", "holdout_weeks")


@dataclass(frozen=True, slots=True)
class LoadedData:
    database: RelationalDatabase
    event_log: EventLog
    data_end: pd.Timestamp


def load_data(*, settings: DataSettings) -> LoadedData:
    database = ParquetDatabaseLoader(db_dir=settings.db_dir).load()
    return LoadedData(
        database=database,
        event_log=build_event_log(database=database),
        data_end=database.latest_row_time(),
    )


def build_feature_blocks(*, settings: FeatureSettings) -> list[FeatureBlock]:
    return [
        ActivityHistoryBlock(settings=settings),
        ProfileBlock(settings=settings),
        RepositoryContextBlock(settings=settings),
        CollaborationBlock(settings=settings),
        BehaviorBlock(settings=settings),
    ]


def build_tuner(*, settings: ExperimentSettings) -> ThresholdTuner:
    return ThresholdTuner(
        grid=settings.model.threshold_grid,
        passes=settings.model.threshold_passes,
    )


def build_classifier_factory(*, settings: ExperimentSettings) -> Callable[[str], EventClassifier]:
    from src.models.xgboost_classifier import XGBoostEventClassifier

    def create(event: str) -> EventClassifier:
        return XGBoostEventClassifier(
            settings=settings.model,
            event=event,
        )

    return create


def build_model(
    *,
    settings: ExperimentSettings,
    data: LoadedData,
) -> ActivityModel:
    if settings.model_type == XGBOOST_MODEL:
        from src.models.tabular_model import TabularEventModel

        return TabularEventModel(classifier_factory=build_classifier_factory(settings=settings))
    if settings.model_type == GNN_MODEL:
        from src.models.gnn_model import RelationalGnnModel
        from src.services.graph_snapshot import GraphSnapshotBuilder

        return RelationalGnnModel(
            settings=settings.gnn,
            snapshot_builder=GraphSnapshotBuilder(
                database=data.database,
                event_log=data.event_log,
                settings=settings.gnn,
            ),
            tuner=build_tuner(settings=settings),
        )
    if settings.model_type == TABPFN_MODEL:
        from src.models.tabpfn_model import TabPfnEventModel

        return TabPfnEventModel(settings=settings.tabpfn)
    raise ConfigError(f"Unknown model_type {settings.model_type!r}. Use one of {MODEL_TYPES}.")


def build_dataset_builder(
    *,
    settings: ExperimentSettings,
    data: LoadedData,
    cache_dir: Path | None,
) -> DatasetBuilder:
    assembler = FeatureAssembler(blocks=build_feature_blocks(settings=settings.features))
    cache_key = build_cache_key(
        settings={
            "sampling": {
                key: value
                for key, value in settings_to_dict(settings=settings.sampling).items()
                if key not in SPLIT_ONLY_SAMPLING_KEYS
            },
            "features": settings_to_dict(settings=settings.features),
        },
        data_end=data.data_end,
        block_names=assembler.block_names,
    )
    return DatasetBuilder(
        database=data.database,
        event_log=data.event_log,
        selector=PopulationSelector(settings=settings.sampling),
        label_builder=LabelBuilder(settings=settings.sampling),
        assembler=assembler,
        cache_dir=cache_dir,
        cache_key=cache_key,
    )


def build_trainer(
    *,
    settings: ExperimentSettings,
    data: LoadedData,
    paths: RunPaths,
    cache_dir: Path | None,
) -> ExperimentTrainer:
    return ExperimentTrainer(
        settings=settings,
        planner=SamplePlanner(settings=settings.sampling),
        dataset_builder=build_dataset_builder(
            settings=settings,
            data=data,
            cache_dir=cache_dir,
        ),
        model=build_model(
            settings=settings,
            data=data,
        ),
        tuner=build_tuner(settings=settings),
        paths=paths,
    )


def build_predictor(
    *,
    settings: ExperimentSettings,
    data: LoadedData,
    paths: RunPaths,
) -> SubmissionPredictor:
    return SubmissionPredictor(
        data_settings=settings.data,
        database=data.database,
        planner=SamplePlanner(settings=settings.sampling),
        dataset_builder=build_dataset_builder(
            settings=settings,
            data=data,
            cache_dir=None,
        ),
        model=build_model(
            settings=settings,
            data=data,
        ),
        paths=paths,
    )
