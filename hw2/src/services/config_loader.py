from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

from src.data.classes.settings import (
    DataSettings,
    ExperimentSettings,
    FeatureSettings,
    GnnSettings,
    GraphSettings,
    ModelSettings,
    SamplingSettings,
    TabPfnSettings,
)

SettingsT = TypeVar("SettingsT")
NESTED_SECTIONS = {
    "data": DataSettings,
    "sampling": SamplingSettings,
    "features": FeatureSettings,
    "model": ModelSettings,
    "gnn": GnnSettings,
    "tabpfn": TabPfnSettings,
}


class ConfigError(ValueError):
    pass


def expand_path(*, value: str) -> Path:
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ConfigError(f"Unresolved environment variable in path {value!r}. Check the .env file.")
    return Path(expanded)


def build_section(
    *,
    section_type: type[SettingsT],
    values: Mapping[str, Any],
    source: Path,
) -> SettingsT:
    known = {item.name for item in fields(section_type)}
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"Unknown keys in {source} for {section_type.__name__}: {sorted(unknown)}")
    hints = get_type_hints(section_type)
    converted = {}
    for key, value in values.items():
        hint = str(hints[key])
        if key in NESTED_SECTIONS and isinstance(value, Mapping):
            converted[key] = build_section(
                section_type=NESTED_SECTIONS[key],
                values=value,
                source=source,
            )
        elif "Path" in hint:
            converted[key] = expand_path(value=str(value))
        elif "tuple" in hint and isinstance(value, list):
            converted[key] = tuple(value)
        else:
            converted[key] = value
    try:
        return section_type(**converted)
    except TypeError as error:
        raise ConfigError(f"Invalid config {source} for {section_type.__name__}: {error}") from error


def read_yaml(*, path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_experiment_settings(*, path: Path) -> ExperimentSettings:
    return build_section(
        section_type=ExperimentSettings,
        values=read_yaml(path=path),
        source=path,
    )


def load_graph_settings(*, path: Path) -> GraphSettings:
    return build_section(
        section_type=GraphSettings,
        values=read_yaml(path=path),
        source=path,
    )


def settings_to_dict(*, settings: object) -> dict[str, Any]:
    if not is_dataclass(settings):
        raise ConfigError(f"Expected a dataclass, got {type(settings).__name__}")
    result = {}
    for item in fields(settings):
        value = getattr(settings, item.name)
        if is_dataclass(value):
            result[item.name] = settings_to_dict(settings=value)
        elif isinstance(value, Path):
            result[item.name] = str(value)
        elif isinstance(value, tuple):
            result[item.name] = list(value)
        else:
            result[item.name] = value
    return result
