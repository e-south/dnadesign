"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/dataset_registry_modes.py

Registry-mode handlers for validating overlay metadata and schemas.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SchemaError
from .overlays import overlay_metadata, overlay_schema
from .registry import load_registry, load_registry_file, registry_hash, validate_overlay_schema

OverlayRegistryValidator = Callable[[dict[str, Any]], None]
AllowedHashesResolver = Callable[[Any], set[str]]
RegistryValidationRunner = Callable[[Any, OverlayRegistryValidator], None]


@dataclass(frozen=True)
class RegistryModeHandler:
    allowed_hashes: AllowedHashesResolver
    validate_with_registries: RegistryValidationRunner


_REGISTRY_MODE_HANDLERS: dict[str, RegistryModeHandler] = {}


def _normalize_mode_name(mode: str | None) -> str:
    text = str(mode or "").strip().lower()
    if not text:
        raise SchemaError("registry_mode must be a non-empty string")
    return text


def supported_registry_modes() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY_MODE_HANDLERS))


def register_registry_mode(
    *,
    mode: str,
    allowed_hashes: AllowedHashesResolver,
    validate_with_registries: RegistryValidationRunner,
) -> None:
    mode_name = _normalize_mode_name(mode)
    if mode_name in _REGISTRY_MODE_HANDLERS:
        raise SchemaError(f"registry_mode '{mode_name}' is already registered")
    if not callable(allowed_hashes):
        raise SchemaError("allowed_hashes must be callable")
    if not callable(validate_with_registries):
        raise SchemaError("validate_with_registries must be callable")
    _REGISTRY_MODE_HANDLERS[mode_name] = RegistryModeHandler(
        allowed_hashes=allowed_hashes,
        validate_with_registries=validate_with_registries,
    )


def normalize_registry_mode(registry_mode: str | None) -> str:
    mode_name = _normalize_mode_name(registry_mode)
    if mode_name not in _REGISTRY_MODE_HANDLERS:
        raise SchemaError(f"Unsupported registry_mode '{registry_mode}'.")
    return mode_name


def validate_overlays_for_registry_mode(
    *,
    dataset: Any,
    overlays: list[Path],
    mode: str,
    reserved_namespaces: set[str],
) -> None:
    mode_name = normalize_registry_mode(mode)
    handler = _REGISTRY_MODE_HANDLERS[mode_name]
    allowed_hashes = handler.allowed_hashes(dataset)

    def _validate_overlays(registry: dict[str, Any]) -> None:
        for path in overlays:
            meta = overlay_metadata(path)
            key = meta.get("key")
            if not key:
                raise SchemaError(f"Overlay missing required metadata key: {path}")
            ns = meta.get("namespace") or path.stem
            reg_hash = meta.get("registry_hash")
            if reg_hash is None:
                raise SchemaError(f"Overlay missing registry_hash metadata: {path}")
            if reg_hash not in allowed_hashes:
                allowed = ", ".join(sorted(allowed_hashes))
                raise SchemaError(f"Overlay registry_hash mismatch for {path}: {reg_hash} not in [{allowed}].")
            if ns in reserved_namespaces:
                continue
            schema = overlay_schema(path)
            validate_overlay_schema(ns, schema, registry=registry, key=key)

    handler.validate_with_registries(dataset, _validate_overlays)


def _allowed_hashes_current(dataset: Any) -> set[str]:
    return {registry_hash(dataset.root, required=True)}


def _allowed_hashes_frozen(dataset: Any) -> set[str]:
    return {dataset._dataset_registry_hash()}


def _allowed_hashes_either(dataset: Any) -> set[str]:
    allowed_hashes: set[str] = set()
    try:
        allowed_hashes.add(registry_hash(dataset.root, required=True))
    except SchemaError:
        pass
    try:
        allowed_hashes.add(dataset._dataset_registry_hash())
    except SchemaError:
        pass
    if not allowed_hashes:
        raise SchemaError("No registry hash available for overlay validation.")
    return allowed_hashes


def _validate_with_current_registry(dataset: Any, validate: OverlayRegistryValidator) -> None:
    registry = load_registry(dataset.root, required=True)
    validate(registry)


def _validate_with_frozen_registry(dataset: Any, validate: OverlayRegistryValidator) -> None:
    registry = load_registry_file(dataset._frozen_registry_path())
    validate(registry)


def _validate_with_either_registry(dataset: Any, validate: OverlayRegistryValidator) -> None:
    try:
        _validate_with_current_registry(dataset, validate)
    except SchemaError:
        _validate_with_frozen_registry(dataset, validate)


register_registry_mode(
    mode="current",
    allowed_hashes=_allowed_hashes_current,
    validate_with_registries=_validate_with_current_registry,
)
register_registry_mode(
    mode="frozen",
    allowed_hashes=_allowed_hashes_frozen,
    validate_with_registries=_validate_with_frozen_registry,
)
register_registry_mode(
    mode="either",
    allowed_hashes=_allowed_hashes_either,
    validate_with_registries=_validate_with_either_registry,
)

