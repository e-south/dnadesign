"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/transforms.py

Resolve and validate transforms supplied by built-in integrations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core import ContractError, SchemaError, reject_unknown_keys
from .contracts import TransformDescriptor, TransformPathResolver
from .registry import registered_transform, registered_transforms


def transform_descriptors() -> tuple[TransformDescriptor, ...]:
    return registered_transforms()


def transform_names() -> tuple[str, ...]:
    return tuple(descriptor.name for descriptor in registered_transforms())


def transform_descriptor(name: str) -> TransformDescriptor:
    descriptor = registered_transform(name)
    if descriptor is None:
        allowed = ", ".join(transform_names())
        raise SchemaError(f"Unsupported pipeline transform: {name!r}; use module:Class or one of: {allowed}")
    return descriptor


def declared_transform_path_values(name: Any, params: Mapping[str, Any]) -> tuple[Any, ...]:
    descriptor = registered_transform(str(name))
    if descriptor is None:
        return ()
    return tuple(params[key] for key in descriptor.path_params if params.get(key) is not None)


def normalize_transform_config(
    *,
    name: Any,
    params: Mapping[str, Any],
    resolve_path: TransformPathResolver | None = None,
) -> tuple[str, dict[str, Any]]:
    parsed_name = str(name).strip()
    if not parsed_name:
        raise SchemaError("pipeline transform name must be non-empty")
    if ":" in parsed_name:
        return parsed_name, dict(params)

    descriptor = transform_descriptor(parsed_name)
    context = f"pipeline.plugins.{parsed_name}"
    try:
        reject_unknown_keys(params, set(descriptor.allowed_params), context)
        missing = sorted(key for key in descriptor.required_params if params.get(key) is None)
        if missing:
            raise SchemaError(f"{context} missing required parameters: {missing}")

        parsed_params = dict(params)
        if resolve_path is not None:
            for key in descriptor.path_params:
                if parsed_params.get(key) is not None:
                    parsed_params[key] = resolve_path(key, parsed_params[key])
        descriptor.validate_params(parsed_params, context)
        return parsed_name, parsed_params
    except ContractError as exc:
        raise SchemaError(str(exc)) from exc


__all__ = [
    "declared_transform_path_values",
    "normalize_transform_config",
    "transform_descriptor",
    "transform_descriptors",
    "transform_names",
]
