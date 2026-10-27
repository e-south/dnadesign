"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/pipeline/transforms.py

Record transform/plugin loading.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Protocol, Sequence

from ..config import PluginSpec
from ..core import PluginError, Record, SchemaError
from ..integrations import transform_descriptor


class Transform(Protocol):
    def apply(self, record: Record) -> Record: ...


def _load_class(spec: str):
    mod_name, _, obj_name = spec.partition(":")
    if not obj_name:
        raise PluginError(f"Transform spec must be 'module:Class', got: {spec}")
    module = import_module(mod_name)
    cls = getattr(module, obj_name, None)
    if cls is None:
        raise PluginError(f"Could not find transform class '{obj_name}' in module '{mod_name}'")
    return cls


def load_transforms(requested: Sequence[PluginSpec]) -> tuple[Transform, ...]:
    transforms: list[Transform] = []
    for spec in requested:
        name = spec.name
        params = spec.params
        if ":" in name:
            cls = _load_class(name)
            transforms.append(cls(**params))
            continue
        try:
            descriptor = transform_descriptor(name)
        except SchemaError as exc:
            raise PluginError(str(exc)) from exc
        if not isinstance(params, Mapping):
            raise PluginError(f"Transform parameters must be a mapping for {name!r}")
        transforms.append(descriptor.factory(params))
    return tuple(transforms)


def apply_transforms(record: Record, transforms: Sequence[Transform]) -> Record:
    out = record
    for transform in transforms:
        out = transform.apply(out)
    return out
