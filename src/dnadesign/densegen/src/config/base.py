"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/config/base.py

DenseGen config base helpers and strict YAML parsing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

if TYPE_CHECKING:
    from .root import RootConfig


# ---- Strict YAML loader (duplicate keys fail) ----
class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep: bool = False):
    mapping: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise KeyError(f"Duplicate key in YAML: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


LATEST_SCHEMA_VERSION = "2.9"
SUPPORTED_SCHEMA_VERSIONS = {LATEST_SCHEMA_VERSION}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    root: "RootConfig"


def _expand_path(value: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(value))))


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged: dict = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_relative_path(cfg_path: Path, value: str | os.PathLike) -> Path:
    p = _expand_path(value)
    if p.is_absolute():
        return p
    return (cfg_path.parent / p).resolve()


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def resolve_run_root(cfg_path: Path, run_root: str | os.PathLike) -> Path:
    root = resolve_relative_path(cfg_path, run_root)
    if root.exists() and not root.is_dir():
        raise ConfigError(f"densegen.run.root must be a directory: {root}")
    if not root.exists():
        raise ConfigError(f"densegen.run.root does not exist: {root}")
    return root


def resolve_run_scoped_path(cfg_path: Path, run_root: Path, value: str | os.PathLike, *, label: str) -> Path:
    resolved = resolve_relative_path(cfg_path, value)
    if not _is_relative_to(resolved, run_root):
        raise ConfigError(f"{label} must be within densegen.run.root ({run_root}), got: {resolved}")
    return resolved


def resolve_outputs_scoped_path(cfg_path: Path, run_root: Path, value: str | os.PathLike, *, label: str) -> Path:
    resolved = resolve_run_scoped_path(cfg_path, run_root, value, label=label)
    outputs_root = run_root / "outputs"
    if not _is_relative_to(resolved, outputs_root):
        raise ConfigError(f"{label} must be within outputs/ under densegen.run.root ({outputs_root}), got: {resolved}")
    return resolved


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    root: str

    @field_validator("id")
    @classmethod
    def _id_nonempty(cls, v: str):
        value = str(v).strip()
        if not value:
            raise ValueError("run.id must be a non-empty string")
        if "/" in value or "\\" in value:
            raise ValueError("run.id must not contain path separators")
        if value in {".", ".."}:
            raise ValueError("run.id must not be '.' or '..'")
        return value

    @field_validator("root")
    @classmethod
    def _root_nonempty(cls, v: str):
        if not v or not str(v).strip():
            raise ValueError("run.root must be a non-empty string")
        return str(v).strip()
