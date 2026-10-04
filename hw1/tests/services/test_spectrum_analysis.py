from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.services.spectrum_analysis import SpectrumFigureWriter, SpectrumStore, compute_layer_spectrum


def build_spectrum(*, values: list[float], threshold: float = 50.0):
    return compute_layer_spectrum(
        weight=torch.diag(torch.tensor(values)),
        layer_index=0,
        module_name="model.layers.0.self_attn.q_proj",
        energy_threshold_percent=threshold,
    )


def test_cumulative_energy_matches_assignment_hint() -> None:
    torch.manual_seed(0)
    weight = torch.randn(64, 48)
    spectrum = compute_layer_spectrum(
        weight=weight,
        layer_index=3,
        module_name="w",
        energy_threshold_percent=50.0,
    )
    _, singular, _ = torch.linalg.svd(
        weight.float(),
        full_matrices=False,
    )
    expected = torch.cumsum(singular**2, dim=0) / torch.sum(singular**2) * 100
    assert torch.allclose(torch.tensor(spectrum.cumulative_energy_percent), expected)
    assert spectrum.weight_shape == (64, 48)
    assert len(spectrum.singular_values) == 48


@pytest.mark.parametrize(
    ("values", "threshold", "expected_rank"),
    [
        ([2.0, 1.0, 1.0, 1.0, 1.0], 50.0, 1),
        ([1.0, 1.0, 1.0, 1.0], 50.0, 2),
        ([1.0, 1.0, 1.0, 1.0], 51.0, 3),
        ([3.0, 2.0, 1.0], 100.0, 3),
    ],
)
def test_minimum_rank_is_first_index_reaching_threshold(
    values: list[float],
    threshold: float,
    expected_rank: int,
) -> None:
    assert (
        build_spectrum(
            values=values,
            threshold=threshold,
        ).minimum_rank
        == expected_rank
    )


def test_store_and_figures_written(tmp_path: Path) -> None:
    spectrum = build_spectrum(values=[float(value) for value in range(300, 0, -1)])
    store = SpectrumStore(root=tmp_path)
    store.save(spectra=[spectrum])
    writer = SpectrumFigureWriter(
        figure_dir=tmp_path / "figures",
        plot_limit=200,
    )
    written = writer.write(spectra=[spectrum])

    assert store.minimum_rank_path.is_file()
    assert store.spectrum_path(layer_index=0).read_text().count("\n") == 301
    assert sorted(path.suffix for path in written) == [".pdf", ".pdf", ".png", ".png"]
