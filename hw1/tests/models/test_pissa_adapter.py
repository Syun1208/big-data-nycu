from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
import torch.nn as nn

from src.models.pissa_adapter import PiSSADecomposer, PiSSAInjector, verify_trainables
from src.models.pissa_linear import PiSSALinear
from src.models.singular_ranges import TopSingularRange

RANK = 4
SEED = 7
CPU = torch.device("cpu")


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(
            in_features=16,
            out_features=24,
            bias=True,
        )
        self.down_proj = nn.Linear(
            in_features=24,
            out_features=16,
            bias=False,
        )
        self.other = nn.Linear(
            in_features=16,
            out_features=16,
        )


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock(), TinyBlock()])


def build_decomposer() -> PiSSADecomposer:
    return PiSSADecomposer(
        svd_device=CPU,
        range_selectors={"default": TopSingularRange()},
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_decompose_matches_original(
    original_pissa: dict[str, Any],
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(SEED)
    linear = nn.Linear(
        in_features=32,
        out_features=20,
        bias=True,
    ).to(dtype)

    expected, expected_stats = original_pissa["pissa_from_linear_exact"](linear, rank=RANK, component="default")
    actual, actual_stats = build_decomposer().decompose(
        linear=linear,
        rank=RANK,
        component="default",
    )

    for name in ("weight", "pissa_A", "pissa_B", "bias"):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name
    assert actual_stats.relative_error_stored == expected_stats.relative_error_stored

    inputs = torch.randn(3, 32).to(dtype)
    assert torch.equal(actual(inputs), expected(inputs))


def test_unknown_component_rejected() -> None:
    with pytest.raises(ValueError):
        build_decomposer().decompose(
            linear=nn.Linear(
                in_features=8,
                out_features=8,
            ),
            rank=2,
            component="unknown",
        )


def test_inject_and_merge_match_original(original_pissa: dict[str, Any]) -> None:
    torch.manual_seed(SEED)
    base = TinyModel()
    original_model = copy.deepcopy(base)
    refactored_model = copy.deepcopy(base)
    original_pissa["TARGET_MODULE_NAMES"] = {"q_proj", "down_proj"}

    original_pissa["inject_pissa"](original_model, rank=RANK, component="default")
    injector = PiSSAInjector(
        decomposer=build_decomposer(),
        target_module_names=("q_proj", "down_proj"),
    )
    injector.inject(
        model=refactored_model,
        rank=RANK,
        component="default",
    )

    assert [type(module).__name__ for module in refactored_model.modules()] == [
        type(module).__name__ for module in original_model.modules()
    ]
    assert verify_trainables(model=refactored_model) == original_pissa["verify_trainables"](original_model)

    original_pissa["merge_pissa_and_unload"](original_model)
    injector.merge_and_unload(model=refactored_model)

    assert not any(isinstance(module, PiSSALinear) for module in refactored_model.modules())
    original_state = original_model.state_dict()
    for name, tensor in refactored_model.state_dict().items():
        assert torch.equal(tensor, original_state[name]), name
