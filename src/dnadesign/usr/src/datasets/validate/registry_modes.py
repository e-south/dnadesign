"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/datasets/validate/registry_modes.py

Registry-mode handlers for validating overlay metadata and schemas.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import SchemaError
from ...overlays import overlay_metadata, overlay_schema
from ...registry import (
    load_registry,
    load_registry_file,
    namespace_contract_hash_for_entries,
    registry_hash_for_entries,
    validate_overlay_schema,
)

AllowedHashesResolver = Callable[[Any, dict[str, Any], str], set[str]]
OverlayMetadataHashResolver = Callable[[dict[str, Any]], str | None]
OverlayRegistryValidator = Callable[[dict[str, Any]], None]
RegistryValidationRunner = Callable[[Any, OverlayRegistryValidator], None]


@dataclass(frozen=True)
class RegistryModeHandler:
    allowed_hashes: AllowedHashesResolver
    validate_with_registries: RegistryValidationRunner
    overlay_hash_from_metadata: OverlayMetadataHashResolver
    overlay_hash_label: str
    skip_reserved_hash_validation: bool = False


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
    overlay_hash_from_metadata: OverlayMetadataHashResolver,
    overlay_hash_label: str,
    skip_reserved_hash_validation: bool = False,
) -> None:
    mode_name = _normalize_mode_name(mode)
    if mode_name in _REGISTRY_MODE_HANDLERS:
        raise SchemaError(f"registry_mode '{mode_name}' is already registered")
    if not callable(allowed_hashes):
        raise SchemaError("allowed_hashes must be callable")
    if not callable(validate_with_registries):
        raise SchemaError("validate_with_registries must be callable")
    if not callable(overlay_hash_from_metadata):
        raise SchemaError("overlay_hash_from_metadata must be callable")
    if not str(overlay_hash_label or "").strip():
        raise SchemaError("overlay_hash_label must be a non-empty string")
    if not isinstance(skip_reserved_hash_validation, bool):
        raise SchemaError("skip_reserved_hash_validation must be a boolean")
    _REGISTRY_MODE_HANDLERS[mode_name] = RegistryModeHandler(
        allowed_hashes=allowed_hashes,
        validate_with_registries=validate_with_registries,
        overlay_hash_from_metadata=overlay_hash_from_metadata,
        overlay_hash_label=str(overlay_hash_label),
        skip_reserved_hash_validation=skip_reserved_hash_validation,
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

    def _validate_overlays(registry: dict[str, Any]) -> None:
        for path in overlays:
            meta = overlay_metadata(path)
            key = meta.get("key")
            if not key:
                raise SchemaError(f"Overlay missing required metadata key: {path}")
            ns = meta.get("namespace") or path.stem
            if not (ns in reserved_namespaces and handler.skip_reserved_hash_validation):
                reg_hash = handler.overlay_hash_from_metadata(meta)
                if reg_hash is None:
                    raise SchemaError(f"Overlay missing {handler.overlay_hash_label} metadata: {path}")
                allowed_hashes = handler.allowed_hashes(dataset, registry, ns)
                if reg_hash not in allowed_hashes:
                    allowed = ", ".join(sorted(allowed_hashes))
                    raise SchemaError(
                        f"Overlay {handler.overlay_hash_label} mismatch for {path}: {reg_hash} not in [{allowed}]."
                    )
            if ns in reserved_namespaces:
                continue
            schema = overlay_schema(path)
            validate_overlay_schema(ns, schema, registry=registry, key=key)

    handler.validate_with_registries(dataset, _validate_overlays)


def _allowed_hashes_current(dataset: Any, registry: dict[str, Any], namespace: str) -> set[str]:
    _ = (dataset, namespace)
    return {registry_hash_for_entries(registry)}


def _allowed_hashes_frozen(dataset: Any, registry: dict[str, Any], namespace: str) -> set[str]:
    _ = (dataset, namespace)
    return {registry_hash_for_entries(registry)}


def _allowed_hashes_namespace(dataset: Any, registry: dict[str, Any], namespace: str) -> set[str]:
    _ = dataset
    return {namespace_contract_hash_for_entries(registry, namespace)}


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


def _overlay_registry_hash_from_metadata(meta: dict[str, Any]) -> str | None:
    return meta.get("registry_hash")


def _overlay_namespace_contract_hash_from_metadata(meta: dict[str, Any]) -> str | None:
    return meta.get("namespace_contract_hash")


register_registry_mode(
    mode="current",
    allowed_hashes=_allowed_hashes_current,
    validate_with_registries=_validate_with_current_registry,
    overlay_hash_from_metadata=_overlay_registry_hash_from_metadata,
    overlay_hash_label="registry_hash",
)
register_registry_mode(
    mode="frozen",
    allowed_hashes=_allowed_hashes_frozen,
    validate_with_registries=_validate_with_frozen_registry,
    overlay_hash_from_metadata=_overlay_registry_hash_from_metadata,
    overlay_hash_label="registry_hash",
)
register_registry_mode(
    mode="either",
    allowed_hashes=_allowed_hashes_current,
    validate_with_registries=_validate_with_either_registry,
    overlay_hash_from_metadata=_overlay_registry_hash_from_metadata,
    overlay_hash_label="registry_hash",
)
register_registry_mode(
    mode="namespace-current",
    allowed_hashes=_allowed_hashes_namespace,
    validate_with_registries=_validate_with_current_registry,
    overlay_hash_from_metadata=_overlay_namespace_contract_hash_from_metadata,
    overlay_hash_label="namespace_contract_hash",
    skip_reserved_hash_validation=True,
)
register_registry_mode(
    mode="namespace-frozen",
    allowed_hashes=_allowed_hashes_namespace,
    validate_with_registries=_validate_with_frozen_registry,
    overlay_hash_from_metadata=_overlay_namespace_contract_hash_from_metadata,
    overlay_hash_label="namespace_contract_hash",
    skip_reserved_hash_validation=True,
)
register_registry_mode(
    mode="namespace-either",
    allowed_hashes=_allowed_hashes_namespace,
    validate_with_registries=_validate_with_either_registry,
    overlay_hash_from_metadata=_overlay_namespace_contract_hash_from_metadata,
    overlay_hash_label="namespace_contract_hash",
    skip_reserved_hash_validation=True,
)
