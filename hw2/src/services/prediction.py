from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.classes.samples import CutoffSample
from src.data.classes.settings import EVENTS, STUDENT_ID, DataSettings
from src.data.loaders.relational_db import RelationalDatabase
from src.interface.activity_model import ActivityModel
from src.services.dataset_builder import DatasetBuilder
from src.services.evaluation import USER_COLUMN, SubmissionEvaluator
from src.services.preview import preview_samples
from src.services.sampling import SamplePlanner
from src.services.scoring import apply_thresholds
from src.services.training import RunPaths, read_thresholds

logger = logging.getLogger(__name__)

TEST_INPUT_FILE = "test_input.csv"
OUTPUT_FILE = f"output_{STUDENT_ID}.csv"
PROBABILITY_FILE = f"output_{STUDENT_ID}_probabilities.csv"


def read_test_users(
    *,
    data_settings: DataSettings,
    database: RelationalDatabase,
) -> tuple[pd.Series | None, pd.Index | None]:
    path = data_settings.evaluation_dir / TEST_INPUT_FILE
    if not path.is_file():
        logger.warning("%s is not released yet. Predicting for the users picked by the population rule.", path)
        return None, None
    github_ids = pd.read_csv(path)[USER_COLUMN].astype("int64")
    mapped = github_ids.map(database.github_to_user_ids())
    missing = int(mapped.isna().sum())
    if missing:
        logger.warning("%d test users are not in Users.parquet. They get all-zero predictions.", missing)
    logger.info("Test input: %d users from %s", len(github_ids), path)
    return github_ids, pd.Index(mapped.dropna().astype("int64").unique(), name="user_id")


class SubmissionPredictor:
    def __init__(
        self,
        *,
        data_settings: DataSettings,
        database: RelationalDatabase,
        planner: SamplePlanner,
        dataset_builder: DatasetBuilder,
        model: ActivityModel,
        paths: RunPaths,
    ) -> None:
        self._data_settings = data_settings
        self._database = database
        self._planner = planner
        self._builder = dataset_builder
        self._model = model
        self._paths = paths

    def predict(self, *, data_end: pd.Timestamp) -> Path:
        self._model.load(model_dir=self._paths.model_dir)
        thresholds = read_thresholds(model_dir=self._paths.model_dir)
        cutoff = self._planner.prediction_cutoff(data_end=data_end)
        logger.info(
            "🔮 %s: cutoff %s, features until %s, population at %s",
            self._model.name,
            cutoff.label,
            cutoff.feature_time,
            cutoff.population_time,
        )
        github_ids, user_ids = read_test_users(
            data_settings=self._data_settings,
            database=self._database,
        )
        sample = CutoffSample(
            cutoff=cutoff,
            dataset=self._builder.build(
                cutoff=cutoff,
                user_ids=user_ids,
            ),
        )
        preview_samples(
            title="Test data",
            samples=[sample],
        )
        probabilities = self._model.predict_proba(samples=[sample])
        predictions = pd.DataFrame(
            apply_thresholds(
                probabilities=probabilities,
                thresholds=thresholds,
            ),
            columns=list(EVENTS),
        )
        user_column = sample.dataset.meta["github_user_id"].to_numpy()
        predictions.insert(0, USER_COLUMN, user_column)
        if github_ids is not None:
            predictions = (
                predictions.set_index(USER_COLUMN)
                .reindex(github_ids.to_numpy(), fill_value=0)
                .rename_axis(USER_COLUMN)
                .reset_index()
            )
        predictions[list(EVENTS)] = predictions[list(EVENTS)].astype("int64")
        self._paths.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path = self._paths.output_dir / OUTPUT_FILE
        predictions.to_csv(
            output_path,
            index=False,
        )
        probability_frame = pd.DataFrame(probabilities, columns=list(EVENTS))
        probability_frame.insert(0, USER_COLUMN, user_column)
        probability_frame.to_csv(
            self._paths.output_dir / PROBABILITY_FILE,
            index=False,
        )
        logger.info(
            "💾 Wrote %s (%d users). Predicted positive rate: %s",
            output_path,
            len(predictions),
            ", ".join(f"{event}={predictions[event].mean():.2f}" for event in EVENTS),
        )
        SubmissionEvaluator(evaluation_dir=self._data_settings.evaluation_dir).evaluate(
            prediction=output_path,
            truth=None,
        )
        return output_path
