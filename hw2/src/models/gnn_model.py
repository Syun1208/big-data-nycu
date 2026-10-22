from __future__ import annotations

import copy
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.classes.samples import CutoffSample
from src.data.classes.settings import EVENTS, GnnSettings
from src.models.relational_gnn import GraphTensors, RelationalGNN, RelationTensors, encode_edges
from src.services.graph_snapshot import GraphSnapshot, GraphSnapshotBuilder
from src.services.scoring import ThresholdTuner
from src.utils.progress import TRAIN, create_progress

logger = logging.getLogger(__name__)

WEIGHTS_FILE = "gnn.pt"
META_FILE = "gnn_meta.json"
RELATIONS = (
    ("user__repo", "user", "repo"),
    ("user__pr", "user", "pr"),
    ("user__issue", "user", "issue"),
    ("pr__repo", "pr", "repo"),
    ("issue__repo", "issue", "repo"),
    ("issue__pr", "issue", "pr"),
)


class RelationalGnnModel:
    def __init__(
        self,
        *,
        settings: GnnSettings,
        snapshot_builder: GraphSnapshotBuilder,
        tuner: ThresholdTuner,
    ) -> None:
        self._settings = settings
        self._snapshot_builder = snapshot_builder
        self._tuner = tuner
        self._device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
        self._network: RelationalGNN | None = None
        self._columns: list[str] = []
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._history: list[dict] = []

    @property
    def name(self) -> str:
        base = "rdl_gnn_hybrid" if self._settings.use_engineered_user_features else "rdl_gnn"
        return base if self._settings.conv_type == "sage" else f"{base}_{self._settings.conv_type}"

    def fit(
        self,
        *,
        train: Sequence[CutoffSample],
        calibration: Sequence[CutoffSample],
    ) -> None:
        if self._device.type != "cuda":
            raise RuntimeError("CUDA is not available for the GNN. Check CUDA_VISIBLE_DEVICES and the torch build.")
        torch.manual_seed(self._settings.seed)
        np.random.seed(self._settings.seed)
        self._fit_normalizer(samples=train)
        self._network = self._new_network().to(self._device)
        optimizer = torch.optim.AdamW(
            self._network.parameters(),
            lr=self._settings.learning_rate,
            weight_decay=self._settings.weight_decay,
        )
        train_graphs = self._snapshots(samples=train)
        calibration_graphs = self._snapshots(samples=calibration)
        generator = np.random.default_rng(self._settings.seed)
        best_score, best_state, stale = -1.0, None, 0
        with create_progress(
            stage=TRAIN,
            total=self._settings.epochs,
            unit="epoch",
        ) as bar:
            for epoch in range(1, self._settings.epochs + 1):
                train_loss = self._train_epoch(
                    samples=train,
                    graphs=train_graphs,
                    optimizer=optimizer,
                    order=generator.permutation(len(train)),
                )
                probabilities = self._predict(samples=calibration, graphs=calibration_graphs)
                truth = np.concatenate([sample.dataset.labels.to_numpy() for sample in calibration])
                _, report = self._tuner.tune(probabilities=probabilities, truth=truth)
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
        logger.info("🕸️ %s: best calibration score %.4f after %d epochs", self.name, best_score, len(self._history))

    def predict_proba(self, *, samples: Sequence[CutoffSample]) -> np.ndarray:
        return self._predict(
            samples=samples,
            graphs=self._snapshots(samples=samples),
        )

    def feature_importance(self) -> pd.DataFrame | None:
        return None

    def save(self, *, model_dir: Path) -> None:
        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        torch.save(self._network.state_dict(), model_dir / WEIGHTS_FILE)
        meta = {
            "columns": self._columns,
            "mean": None if self._mean is None else self._mean.tolist(),
            "scale": None if self._scale is None else self._scale.tolist(),
            "history": self._history,
        }
        (model_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, *, model_dir: Path) -> None:
        meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
        self._columns = meta["columns"]
        self._mean = None if meta["mean"] is None else np.asarray(meta["mean"], dtype="float32")
        self._scale = None if meta["scale"] is None else np.asarray(meta["scale"], dtype="float32")
        self._history = meta["history"]
        self._network = self._new_network().to(self._device)
        self._network.load_state_dict(torch.load(model_dir / WEIGHTS_FILE, map_location=self._device, weights_only=True))
        logger.info("Loaded %s from %s", self.name, model_dir)

    def _new_network(self) -> RelationalGNN:
        return RelationalGNN(
            hidden_dim=self._settings.hidden_dim,
            num_layers=self._settings.num_layers,
            dropout=self._settings.dropout,
            relations=RELATIONS,
            user_extra_dim=len(self._columns) + 1 if self._settings.use_engineered_user_features else 0,
            conv_type=self._settings.conv_type,
            num_heads=self._settings.num_heads,
        )

    def _fit_normalizer(self, *, samples: Sequence[CutoffSample]) -> None:
        if not self._settings.use_engineered_user_features:
            return
        features = pd.concat([sample.dataset.features for sample in samples], ignore_index=True)
        self._columns = list(features.columns)
        values = np.sign(features.to_numpy(dtype="float32")) * np.log1p(np.abs(features.to_numpy(dtype="float32")))
        self._mean = values.mean(axis=0)
        self._scale = values.std(axis=0) + 1e-6

    def _snapshots(self, *, samples: Sequence[CutoffSample]) -> list[GraphSnapshot]:
        return [self._snapshot_builder.build(feature_time=sample.cutoff.feature_time) for sample in samples]

    def _to_tensors(
        self,
        *,
        snapshot: GraphSnapshot,
        sample: CutoffSample,
    ) -> tuple[GraphTensors, torch.Tensor]:
        device = self._device
        node_features = {name: torch.from_numpy(values) for name, values in snapshot.node_features.items()}
        user_index = torch.from_numpy(sample.dataset.meta["user_id"].to_numpy(dtype="int64"))
        if self._settings.use_engineered_user_features:
            features = sample.dataset.features[self._columns].to_numpy(dtype="float32")
            values = (np.sign(features) * np.log1p(np.abs(features)) - self._mean) / self._scale
            extra = torch.zeros((node_features["user"].shape[0], len(self._columns) + 1))
            extra[user_index, :-1] = torch.from_numpy(values.astype("float32"))
            extra[user_index, -1] = 1.0
            node_features["user"] = torch.cat([node_features["user"], extra], dim=1)
        relations = {
            name: RelationTensors(
                source_type=edges.source_type,
                target_type=edges.target_type,
                source=torch.from_numpy(edges.source).to(device),
                target=torch.from_numpy(edges.target).to(device),
                edge_features=encode_edges(
                    kind=torch.from_numpy(edges.kind).to(device),
                    age_days=torch.from_numpy(edges.age_days).to(device),
                    lookback_days=float(self._settings.lookback_days),
                ),
            )
            for name, edges in snapshot.relations.items()
        }
        graph = GraphTensors(
            node_features={name: values.to(device) for name, values in node_features.items()},
            relations=relations,
        )
        return graph, user_index.to(device)

    def _train_epoch(
        self,
        *,
        samples: Sequence[CutoffSample],
        graphs: Sequence[GraphSnapshot],
        optimizer: torch.optim.Optimizer,
        order: np.ndarray,
    ) -> float:
        self._network.train()
        losses = []
        for index in order:
            graph, user_index = self._to_tensors(snapshot=graphs[index], sample=samples[index])
            labels = torch.from_numpy(samples[index].dataset.labels[list(EVENTS)].to_numpy(dtype="float32")).to(self._device)
            logits = self._network(graph=graph, user_index=user_index)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._network.parameters(), self._settings.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
        return float(np.mean(losses))

    @torch.no_grad()
    def _predict(
        self,
        *,
        samples: Sequence[CutoffSample],
        graphs: Sequence[GraphSnapshot],
    ) -> np.ndarray:
        self._network.eval()
        outputs = []
        for sample, snapshot in zip(samples, graphs):
            graph, user_index = self._to_tensors(snapshot=snapshot, sample=sample)
            outputs.append(torch.sigmoid(self._network(graph=graph, user_index=user_index)).cpu().numpy())
        return np.concatenate(outputs).astype("float64")
