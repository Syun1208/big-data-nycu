from __future__ import annotations

import copy
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.classes.samples import CutoffSample
from src.data.classes.settings import EVENTS
from src.services.scoring import ThresholdTuner
from src.utils.progress import TRAIN, create_progress

logger = logging.getLogger(__name__)

SequenceTensors = tuple[torch.Tensor, ...]


class SequenceTrainingSettings(Protocol):
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    patience: int
    grad_clip: float
    use_engineered_user_features: bool
    device: str
    seed: int


class BatchedSequenceModel(ABC):
    base_name: str = ""
    weights_file: str = ""
    meta_file: str = ""

    def __init__(
        self,
        *,
        settings: SequenceTrainingSettings,
        tuner: ThresholdTuner,
    ) -> None:
        self._settings = settings
        self._tuner = tuner
        self._device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
        self._network: nn.Module | None = None
        self._columns: list[str] = []
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._history: list[dict] = []

    @property
    def name(self) -> str:
        return f"{self.base_name}_hybrid" if self._settings.use_engineered_user_features else self.base_name

    @abstractmethod
    def _new_network(self) -> nn.Module: ...

    @abstractmethod
    def _sequence_tensors(self, *, samples: Sequence[CutoffSample]) -> SequenceTensors: ...

    @abstractmethod
    def _logits(self, *, sequences: SequenceTensors, features: torch.Tensor | None) -> torch.Tensor: ...

    @abstractmethod
    def _embedding(self, *, sequences: SequenceTensors) -> torch.Tensor: ...

    def fit(
        self,
        *,
        train: Sequence[CutoffSample],
        calibration: Sequence[CutoffSample],
    ) -> None:
        if self._device.type != "cuda":
            raise RuntimeError(f"CUDA is not available for {self.name}. Check CUDA_VISIBLE_DEVICES and the torch build.")
        torch.manual_seed(self._settings.seed)
        np.random.seed(self._settings.seed)
        self._fit_normalizer(samples=train)
        self._network = self._new_network().to(self._device)
        optimizer = torch.optim.AdamW(
            self._network.parameters(),
            lr=self._settings.learning_rate,
            weight_decay=self._settings.weight_decay,
        )
        sequences, features = self._inputs(samples=train)
        labels = torch.from_numpy(np.concatenate([sample.dataset.labels[list(EVENTS)].to_numpy(dtype="float32") for sample in train])).to(self._device)
        calibration_inputs = self._inputs(samples=calibration)
        calibration_truth = np.concatenate([sample.dataset.labels[list(EVENTS)].to_numpy() for sample in calibration])
        generator = torch.Generator(device=self._device).manual_seed(self._settings.seed)
        logger.info("⏳ %s train rows %d, calibration rows %d", self.name, len(labels), len(calibration_truth))
        best_score, best_state, stale = -1.0, None, 0
        with create_progress(
            stage=TRAIN,
            total=self._settings.epochs,
            unit="epoch",
        ) as bar:
            for epoch in range(1, self._settings.epochs + 1):
                train_loss = self._train_epoch(
                    sequences=sequences,
                    features=features,
                    labels=labels,
                    optimizer=optimizer,
                    order=torch.randperm(len(labels), device=self._device, generator=generator),
                )
                probabilities = self._predict(inputs=calibration_inputs)
                _, report = self._tuner.tune(probabilities=probabilities, truth=calibration_truth)
                self._history.append({"epoch": epoch, "train_loss": train_loss, "calibration_score": report.score})
                logger.debug("Epoch %d: loss %.4f, calibration score %.4f", epoch, train_loss, report.score)
                bar.set_postfix_str(f"loss {train_loss:.4f} cal {report.score:.4f}")
                bar.update(1)
                if report.score > best_score:
                    best_score, best_state, stale = report.score, copy.deepcopy(self._network.state_dict()), 0
                else:
                    stale += 1
                    if stale >= self._settings.patience:
                        break
        self._network.load_state_dict(best_state)
        logger.info("⏱️ %s: best calibration score %.4f after %d epochs", self.name, best_score, len(self._history))

    def predict_proba(self, *, samples: Sequence[CutoffSample]) -> np.ndarray:
        return self._predict(inputs=self._inputs(samples=samples))

    @torch.no_grad()
    def embed(self, *, samples: Sequence[CutoffSample]) -> np.ndarray:
        self._network.eval()
        sequences, _ = self._inputs(samples=samples)
        outputs = [
            self._embedding(sequences=tuple(tensor[batch] for tensor in sequences)).cpu().numpy()
            for batch in self._batches(size=len(sequences[0]))
        ]
        return np.concatenate(outputs).astype("float32")

    def feature_importance(self) -> pd.DataFrame | None:
        return None

    def save(self, *, model_dir: Path) -> None:
        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        torch.save(self._network.state_dict(), model_dir / self.weights_file)
        meta = {
            "columns": self._columns,
            "mean": None if self._mean is None else self._mean.tolist(),
            "scale": None if self._scale is None else self._scale.tolist(),
            "history": self._history,
        }
        (model_dir / self.meta_file).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, *, model_dir: Path) -> None:
        meta = json.loads((model_dir / self.meta_file).read_text(encoding="utf-8"))
        self._columns = meta["columns"]
        self._mean = None if meta["mean"] is None else np.asarray(meta["mean"], dtype="float32")
        self._scale = None if meta["scale"] is None else np.asarray(meta["scale"], dtype="float32")
        self._history = meta["history"]
        self._network = self._new_network().to(self._device)
        self._network.load_state_dict(torch.load(model_dir / self.weights_file, map_location=self._device, weights_only=True))
        logger.info("Loaded %s from %s", self.name, model_dir)

    def _fit_normalizer(self, *, samples: Sequence[CutoffSample]) -> None:
        if not self._settings.use_engineered_user_features:
            return
        features = pd.concat([sample.dataset.features for sample in samples], ignore_index=True)
        self._columns = list(features.columns)
        values = np.sign(features.to_numpy(dtype="float32")) * np.log1p(np.abs(features.to_numpy(dtype="float32")))
        self._mean = values.mean(axis=0)
        self._scale = values.std(axis=0) + 1e-6

    def _inputs(self, *, samples: Sequence[CutoffSample]) -> tuple[SequenceTensors, torch.Tensor | None]:
        sequences = self._sequence_tensors(samples=samples)
        if not self._columns:
            return sequences, None
        features = pd.concat([sample.dataset.features for sample in samples], ignore_index=True)[self._columns].to_numpy(dtype="float32")
        values = (np.sign(features) * np.log1p(np.abs(features)) - self._mean) / self._scale
        return sequences, torch.from_numpy(values.astype("float32")).to(self._device)

    def _batches(self, *, size: int) -> list[slice]:
        step = self._settings.batch_size
        return [slice(start, start + step) for start in range(0, size, step)]

    def _train_epoch(
        self,
        *,
        sequences: SequenceTensors,
        features: torch.Tensor | None,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        order: torch.Tensor,
    ) -> float:
        self._network.train()
        losses = []
        for batch in self._batches(size=len(order)):
            rows = order[batch]
            logits = self._logits(
                sequences=tuple(tensor[rows] for tensor in sequences),
                features=None if features is None else features[rows],
            )
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._network.parameters(), self._settings.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
        return float(np.mean(losses))

    @torch.no_grad()
    def _predict(self, *, inputs: tuple[SequenceTensors, torch.Tensor | None]) -> np.ndarray:
        self._network.eval()
        sequences, features = inputs
        outputs = [
            torch.sigmoid(
                self._logits(
                    sequences=tuple(tensor[batch] for tensor in sequences),
                    features=None if features is None else features[batch],
                )
            ).cpu().numpy()
            for batch in self._batches(size=len(sequences[0]))
        ]
        return np.concatenate(outputs).astype("float64")
