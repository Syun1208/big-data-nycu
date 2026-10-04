from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.pissa_adapter import PiSSADecomposer
from src.models.singular_ranges import BottomSingularRange, PositionSingularRange, TopSingularRange
from src.services.factory import build_range_selectors

SINGULAR_COUNT = 2048
RANK = 16


@pytest.mark.parametrize(
    ("component", "expected_start"),
    [
        ("default", 0),
        ("p25", SINGULAR_COUNT // 4),
        ("p50", SINGULAR_COUNT // 2),
        ("bottom", SINGULAR_COUNT - RANK),
    ],
)
def test_component_start_positions(
    component: str,
    expected_start: int,
) -> None:
    selector = build_range_selectors()[component]
    assert (
        selector.select_start(
            singular_count=SINGULAR_COUNT,
            rank=RANK,
        )
        == expected_start
    )


def test_position_range_rejects_overflow() -> None:
    with pytest.raises(ValueError):
        PositionSingularRange(
            numerator=3,
            denominator=4,
        ).select_start(
            singular_count=8,
            rank=4,
        )


def test_decompose_uses_selected_singular_values() -> None:
    singular_values = torch.tensor([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    weight = torch.diag(singular_values)
    linear = nn.Linear(
        in_features=8,
        out_features=8,
        bias=False,
    )
    with torch.no_grad():
        linear.weight.copy_(weight)

    decomposer = PiSSADecomposer(
        svd_device=torch.device("cpu"),
        range_selectors={
            "default": TopSingularRange(),
            "p50": PositionSingularRange(
                numerator=1,
                denominator=2,
            ),
            "bottom": BottomSingularRange(),
        },
    )
    expected = {"default": [8.0, 7.0], "p50": [4.0, 3.0], "bottom": [2.0, 1.0]}
    for component, values in expected.items():
        layer, _ = decomposer.decompose(
            linear=linear,
            rank=2,
            component=component,
        )
        adapter = layer.pissa_A @ layer.pissa_B
        assert torch.allclose(
            torch.linalg.svdvals(adapter)[:2],
            torch.tensor(values),
            atol=1e-5,
        ), component
        assert torch.allclose(
            layer.weight + adapter,
            weight,
            atol=1e-5,
        )
