from __future__ import annotations

import logging
from collections.abc import Collection, Mapping

import torch
import torch.nn as nn

from src.data.classes.records import PiSSAInitStats
from src.interface.singular_range import SingularRangeSelector
from src.models.pissa_linear import PiSSALinear
from src.utils.cuda_memory import release_cuda_memory
from src.utils.progress import Stage, create_progress

logger = logging.getLogger(__name__)

TARGET_MODULE_NAMES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)
TRAINABLE_SUFFIXES = ("pissa_A", "pissa_B")
INJECT = Stage(emoji="🧬", title="PiSSA SVD init", colour="blue")
MERGE = Stage(emoji="🔗", title="PiSSA merge", colour="magenta")


class PiSSADecomposer:
    def __init__(
        self,
        *,
        svd_device: torch.device,
        range_selectors: Mapping[str, SingularRangeSelector],
    ) -> None:
        self._svd_device = svd_device
        self._range_selectors = dict(range_selectors)

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self._range_selectors)

    @torch.no_grad()
    def decompose(
        self,
        *,
        linear: nn.Linear,
        rank: int,
        component: str,
        layer_name: str = "",
    ) -> tuple[PiSSALinear, PiSSAInitStats]:
        if component not in self._range_selectors:
            raise ValueError(component)

        weight = linear.weight.detach().to(
            self._svd_device,
            dtype=torch.float32,
        )
        max_rank = min(weight.shape)
        if rank > max_rank:
            raise ValueError(f"{layer_name}: rank={rank} > max rank={max_rank}")

        left, singular, right = torch.linalg.svd(
            weight,
            full_matrices=False,
        )
        start = self._range_selectors[component].select_start(
            singular_count=len(singular),
            rank=rank,
        )
        end = start + rank

        sqrt_singular = torch.sqrt(singular[start:end])
        pissa_A = left[:, start:end] * sqrt_singular.unsqueeze(0)
        pissa_B = sqrt_singular.unsqueeze(1) * right[start:end, :]
        residual = weight - (pissa_A @ pissa_B)

        stored = residual.to(linear.weight.dtype).cpu()
        pissa_A_cpu, pissa_B_cpu, weight_cpu = pissa_A.cpu(), pissa_B.cpu(), weight.cpu()
        relative_error = (
            torch.linalg.vector_norm(weight_cpu - (stored.float() + pissa_A_cpu @ pissa_B_cpu))
            / torch.linalg.vector_norm(weight_cpu)
        ).item()

        replacement = PiSSALinear(
            weight_residual=stored,
            pissa_A=pissa_A_cpu,
            pissa_B=pissa_B_cpu,
            bias=None if linear.bias is None else linear.bias.detach().cpu(),
        )
        stats = PiSSAInitStats(
            name=layer_name,
            rank=rank,
            component=component,
            relative_error_stored=relative_error,
        )

        del weight, left, singular, right, sqrt_singular, pissa_A, pissa_B, residual
        release_cuda_memory()
        return replacement, stats


class PiSSAInjector:
    def __init__(
        self,
        *,
        decomposer: PiSSADecomposer,
        target_module_names: Collection[str] = TARGET_MODULE_NAMES,
    ) -> None:
        self._decomposer = decomposer
        self._target_module_names = frozenset(target_module_names)

    def inject(
        self,
        *,
        model: nn.Module,
        rank: int,
        component: str,
    ) -> list[PiSSAInitStats]:
        freeze_model(model=model)
        targets = self._collect_target_linears(model=model)
        stats = []

        with create_progress(
            stage=INJECT,
            total=len(targets),
            unit="layer",
        ) as bar:
            for name, linear in targets:
                replacement, layer_stats = self._decomposer.decompose(
                    linear=linear,
                    rank=rank,
                    component=component,
                    layer_name=name,
                )
                replace_submodule(
                    root=model,
                    full_name=name,
                    replacement=replacement,
                )
                stats.append(layer_stats)
                logger.debug("Injected PiSSA into %s (err=%.3e)", name, layer_stats.relative_error_stored)
                bar.set_postfix_str(name)
                bar.update(1)
        return stats

    @torch.no_grad()
    def merge_and_unload(self, *, model: nn.Module) -> list[str]:
        targets = [(name, module) for name, module in model.named_modules() if isinstance(module, PiSSALinear)]
        if not targets:
            raise RuntimeError("No PiSSALinear modules found to merge.")

        merged_names = []
        with create_progress(
            stage=MERGE,
            total=len(targets),
            unit="layer",
        ) as bar:
            for name, layer in targets:
                replace_submodule(
                    root=model,
                    full_name=name,
                    replacement=_merge_layer(layer=layer),
                )
                merged_names.append(name)
                bar.update(1)

        remaining = [name for name, module in model.named_modules() if isinstance(module, PiSSALinear)]
        if remaining:
            raise RuntimeError("PiSSA merge incomplete; remaining modules: " + ", ".join(remaining))

        logger.info("🔗 Merged %d PiSSA layers into standard nn.Linear layers.", len(merged_names))
        return merged_names

    def _collect_target_linears(self, *, model: nn.Module) -> list[tuple[str, nn.Linear]]:
        return [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in self._target_module_names
        ]


def _merge_layer(*, layer: PiSSALinear) -> nn.Linear:
    merged_weight = (layer.weight.float() + layer.pissa_A.float() @ layer.pissa_B.float()).to(
        dtype=layer.weight.dtype,
    )
    replacement = nn.Linear(
        in_features=layer.weight.shape[1],
        out_features=layer.weight.shape[0],
        bias=layer.bias is not None,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )
    replacement.weight.copy_(merged_weight)
    replacement.weight.requires_grad_(False)

    if layer.bias is not None:
        replacement.bias.copy_(
            layer.bias.to(
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
        )
        replacement.bias.requires_grad_(False)
    return replacement


def freeze_model(*, model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def replace_submodule(
    *,
    root: nn.Module,
    full_name: str,
    replacement: nn.Module,
) -> None:
    if "." not in full_name:
        setattr(root, full_name, replacement)
        return
    parent_name, child_name = full_name.rsplit(".", 1)
    setattr(root.get_submodule(parent_name), child_name, replacement)


def verify_trainables(*, model: nn.Module) -> tuple[int, int]:
    trainable, total, unexpected = 0, 0, []
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
            if not name.endswith(TRAINABLE_SUFFIXES):
                unexpected.append(name)

    if trainable == 0:
        raise RuntimeError("No trainable PiSSA parameters found.")
    if unexpected:
        raise RuntimeError("Unexpected trainables: " + ", ".join(unexpected))

    logger.info("🧠 Trainable parameters: %s (%.3f%%)", f"{trainable:,}", 100 * trainable / total)
    return trainable, total


def has_pissa_layers(*, model: nn.Module) -> bool:
    return any(isinstance(module, PiSSALinear) for module in model.modules())
