from __future__ import annotations

import csv
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src.data.classes.records import LayerSpectrum
from src.models.loading import PretrainedModelLoader, release_model
from src.utils.plotting import (
    ACADEMIC_MARKERS,
    ACADEMIC_PALETTE,
    apply_academic_style,
    plot_cumulative_energy,
    plot_singular_values,
    save_figure,
)
from src.utils.progress import Stage, create_progress

logger = logging.getLogger(__name__)

ANALYSE = Stage(emoji="🧬", title="SVD q_proj", colour="blue")
PLOT = Stage(emoji="📊", title="Figures", colour="magenta")
PERCENT_SCALE = 100.0
LAYER_COLOURS = ACADEMIC_PALETTE[2:5]
LAYER_MARKERS = ACADEMIC_MARKERS[2:5]


@dataclass(frozen=True, slots=True)
class SpectrumSettings:
    output_dir: Path
    layer_indices: tuple[int, ...] = (0, 7, 15)
    module_path: str = "self_attn.q_proj"
    energy_threshold_percent: float = 50.0
    plot_limit: int = 200


def compute_layer_spectrum(
    *,
    weight: torch.Tensor,
    layer_index: int,
    module_name: str,
    energy_threshold_percent: float,
) -> LayerSpectrum:
    _, singular_values, _ = torch.linalg.svd(
        weight.float(),
        full_matrices=False,
    )
    energy = singular_values**2
    cumulative_energy = torch.cumsum(energy, dim=0) / torch.sum(energy) * PERCENT_SCALE
    reached = torch.nonzero(cumulative_energy >= energy_threshold_percent)
    minimum_rank = int(reached[0].item()) + 1 if len(reached) else len(singular_values)
    return LayerSpectrum(
        layer_index=layer_index,
        module_name=module_name,
        weight_shape=(int(weight.shape[0]), int(weight.shape[1])),
        singular_values=tuple(singular_values.tolist()),
        cumulative_energy_percent=tuple(cumulative_energy.tolist()),
        energy_threshold_percent=energy_threshold_percent,
        minimum_rank=minimum_rank,
    )


class SpectrumStore:
    def __init__(self, *, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def minimum_rank_path(self) -> Path:
        return self._root / "q4_minimum_rank.csv"

    @property
    def summary_path(self) -> Path:
        return self._root / "q4_spectrum_summary.json"

    def spectrum_path(self, *, layer_index: int) -> Path:
        return self._root / f"q4_spectrum_layer{layer_index}.csv"

    def save(self, *, spectra: Sequence[LayerSpectrum]) -> None:
        with self.minimum_rank_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["layer", "module", "rows", "cols", "full_rank", "energy_threshold_percent", "minimum_rank"])
            for spectrum in spectra:
                writer.writerow(
                    [
                        spectrum.layer_index,
                        spectrum.module_name,
                        *spectrum.weight_shape,
                        len(spectrum.singular_values),
                        spectrum.energy_threshold_percent,
                        spectrum.minimum_rank,
                    ]
                )

        for spectrum in spectra:
            with self.spectrum_path(layer_index=spectrum.layer_index).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["index", "singular_value", "cumulative_energy_percent"])
                for index, (value, energy) in enumerate(
                    zip(spectrum.singular_values, spectrum.cumulative_energy_percent),
                    1,
                ):
                    writer.writerow([index, value, energy])

        summary = [
            {key: value for key, value in asdict(spectrum).items() if key not in ("singular_values", "cumulative_energy_percent")}
            for spectrum in spectra
        ]
        self.summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        logger.info("💾 Minimum ranks: %s", self.minimum_rank_path)


class SpectrumFigureWriter:
    def __init__(
        self,
        *,
        figure_dir: Path,
        plot_limit: int,
    ) -> None:
        self._figure_dir = figure_dir
        self._plot_limit = plot_limit

    def singular_value_path(self, *, layer_index: int) -> Path:
        return self._figure_dir / f"q4_singular_values_layer{layer_index}"

    def cumulative_energy_path(self, *, layer_index: int) -> Path:
        return self._figure_dir / f"q4_cumulative_energy_layer{layer_index}"

    def write(self, *, spectra: Sequence[LayerSpectrum]) -> list[Path]:
        apply_academic_style()
        written = []
        with create_progress(
            stage=PLOT,
            total=2 * len(spectra),
            unit="fig",
        ) as bar:
            for position, spectrum in enumerate(spectra):
                colour = LAYER_COLOURS[position % len(LAYER_COLOURS)]
                label = f"Layer {spectrum.layer_index} q_proj"
                written += save_figure(
                    figure=plot_singular_values(
                        singular_values=spectrum.singular_values[: self._plot_limit],
                        colour=colour,
                        label=label,
                    ),
                    path=self.singular_value_path(layer_index=spectrum.layer_index),
                )
                bar.update(1)
                written += save_figure(
                    figure=plot_cumulative_energy(
                        cumulative_energy_percent=spectrum.cumulative_energy_percent[: self._plot_limit],
                        threshold_percent=spectrum.energy_threshold_percent,
                        minimum_rank=spectrum.minimum_rank,
                        colour=colour,
                        marker=LAYER_MARKERS[position % len(LAYER_MARKERS)],
                        label=label,
                    ),
                    path=self.cumulative_energy_path(layer_index=spectrum.layer_index),
                )
                bar.update(1)
        logger.info("📊 Figures saved to %s", self._figure_dir)
        return written


class QProjSpectrumService:
    def __init__(
        self,
        *,
        settings: SpectrumSettings,
        model_loader: PretrainedModelLoader,
        store: SpectrumStore,
        figure_writer: SpectrumFigureWriter,
    ) -> None:
        self._settings = settings
        self._model_loader = model_loader
        self._store = store
        self._figure_writer = figure_writer

    def run(self) -> list[LayerSpectrum]:
        model = self._model_loader.load_model(to_device=False)
        try:
            spectra = self._analyse(model=model)
        finally:
            release_model(model=model)

        self._store.save(spectra=spectra)
        self._figure_writer.write(spectra=spectra)
        return spectra

    @torch.no_grad()
    def _analyse(self, *, model: torch.nn.Module) -> list[LayerSpectrum]:
        spectra = []
        with create_progress(
            stage=ANALYSE,
            total=len(self._settings.layer_indices),
            unit="layer",
        ) as bar:
            for layer_index in self._settings.layer_indices:
                module_name = f"model.layers.{layer_index}.{self._settings.module_path}"
                weight = model.get_submodule(module_name).weight.detach()
                spectrum = compute_layer_spectrum(
                    weight=weight,
                    layer_index=layer_index,
                    module_name=module_name,
                    energy_threshold_percent=self._settings.energy_threshold_percent,
                )
                logger.info(
                    "🧬 Layer %d %s shape=%s: minimum rank for >= %.0f%% energy = %d / %d",
                    layer_index,
                    self._settings.module_path,
                    spectrum.weight_shape,
                    spectrum.energy_threshold_percent,
                    spectrum.minimum_rank,
                    len(spectrum.singular_values),
                )
                spectra.append(spectrum)
                bar.set_postfix_str(f"layer {layer_index}")
                bar.update(1)
        return spectra


def build_spectrum_service(
    *,
    settings: SpectrumSettings,
    model_id: str,
) -> QProjSpectrumService:
    return QProjSpectrumService(
        settings=settings,
        model_loader=PretrainedModelLoader(
            model_id=model_id,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
        store=SpectrumStore(root=settings.output_dir),
        figure_writer=SpectrumFigureWriter(
            figure_dir=settings.output_dir / "figures",
            plot_limit=settings.plot_limit,
        ),
    )
