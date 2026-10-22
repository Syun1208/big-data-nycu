from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

EVENTS = ("open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push")
STUDENT_ID = "314540080"


@dataclass(frozen=True, slots=True)
class DataSettings:
    dataset_dir: Path
    repository_url: str = "https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git"

    @property
    def db_dir(self) -> Path:
        return self.dataset_dir / "data" / "db"

    @property
    def evaluation_dir(self) -> Path:
        return self.dataset_dir / "evaluation"


@dataclass(frozen=True, slots=True)
class SamplingSettings:
    anchor_cutoff: str = "2026-11-11 16:00:00"
    train_start: str = "2026-06-03 16:00:00"
    horizon_days: int = 7
    feature_gap_hours: int = 4
    population_lag_days: int = 14
    population_window_days: int = 90
    population_min_records: int = 5
    calibration_weeks: int = 1
    holdout_weeks: int = 1

    @property
    def anchor(self) -> pd.Timestamp:
        return pd.Timestamp(self.anchor_cutoff)

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_start)

    @property
    def horizon(self) -> pd.Timedelta:
        return pd.Timedelta(days=self.horizon_days)

    @property
    def feature_gap(self) -> pd.Timedelta:
        return pd.Timedelta(hours=self.feature_gap_hours)


@dataclass(frozen=True, slots=True)
class FeatureSettings:
    windows_days: tuple[int, ...] = (7, 14, 28, 56, 91)
    half_lives_days: tuple[float, ...] = (3.0, 14.0)
    active_weeks: int = 8
    recency_cap_days: float = 365.0
    repo_lookback_days: int = 28
    repo_heat_days: int = 7
    collaboration_days: int = 28
    received_days: int = 14
    open_item_days: int = 56
    behavior_weeks: int = 13


@dataclass(frozen=True, slots=True)
class ModelSettings:
    n_estimators: int = 2000
    learning_rate: float = 0.03
    max_depth: int = 6
    min_child_weight: float = 5.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 100
    device: str = "cuda"
    seed: int = 42
    threshold_grid: tuple[float, ...] = tuple(round(0.02 * step, 2) for step in range(1, 50))
    threshold_passes: int = 3


@dataclass(frozen=True, slots=True)
class GnnSettings:
    lookback_days: int = 91
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    epochs: int = 40
    patience: int = 8
    grad_clip: float = 1.0
    use_engineered_user_features: bool = False
    conv_type: str = "sage"
    num_heads: int = 4
    device: str = "cuda"
    seed: int = 42


@dataclass(frozen=True, slots=True)
class TabPfnSettings:
    model_version: str = "v2"
    n_estimators: int = 4
    max_context_rows: int = 10000
    positive_fraction: float = 0.25
    recent_weeks: int = 0
    predict_batch_rows: int = 2000
    device: str = "cuda"
    seed: int = 42


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    name: str
    data: DataSettings
    model_type: str = "xgboost"
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    gnn: GnnSettings = field(default_factory=GnnSettings)
    tabpfn: TabPfnSettings = field(default_factory=TabPfnSettings)


@dataclass(frozen=True, slots=True)
class GraphSettings:
    name: str
    data: DataSettings
    github_user_ids: tuple[int, ...] = (22996444, 28983449)
    start_time: str = "2026-09-15 00:00:00"
    end_time: str = "2026-09-17 00:00:00"
    max_text_length: int = 28
