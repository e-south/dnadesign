"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_dataset_registry_modes_module.py

Tests for dataset registry-mode handler registration and normalization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.usr.src.dataset_registry_modes import normalize_registry_mode, register_registry_mode
from dnadesign.usr.src.errors import SchemaError


def test_normalize_registry_mode_accepts_builtin_modes() -> None:
    assert normalize_registry_mode("current") == "current"
    assert normalize_registry_mode("frozen") == "frozen"
    assert normalize_registry_mode("either") == "either"
    assert normalize_registry_mode("namespace-current") == "namespace-current"
    assert normalize_registry_mode("namespace-frozen") == "namespace-frozen"
    assert normalize_registry_mode("namespace-either") == "namespace-either"


def test_normalize_registry_mode_rejects_unknown_mode() -> None:
    with pytest.raises(SchemaError, match="Unsupported registry_mode"):
        normalize_registry_mode("mystery")


def test_register_registry_mode_rejects_duplicate_name() -> None:
    mode_name = "unit_custom_mode"
    register_registry_mode(
        mode=mode_name,
        allowed_hashes=lambda dataset, registry, namespace: set(),
        validate_with_registries=lambda dataset, validate: None,
        overlay_hash_from_metadata=lambda meta: None,
        overlay_hash_label="registry_hash",
    )
    with pytest.raises(SchemaError, match=f"registry_mode '{mode_name}' is already registered"):
        register_registry_mode(
            mode=mode_name,
            allowed_hashes=lambda dataset, registry, namespace: set(),
            validate_with_registries=lambda dataset, validate: None,
            overlay_hash_from_metadata=lambda meta: None,
            overlay_hash_label="registry_hash",
        )


def test_register_registry_mode_rejects_non_boolean_reserved_hash_flag() -> None:
    with pytest.raises(SchemaError, match="skip_reserved_hash_validation must be a boolean"):
        register_registry_mode(
            mode="unit_invalid_reserved_hash_flag",
            allowed_hashes=lambda dataset, registry, namespace: set(),
            validate_with_registries=lambda dataset, validate: None,
            overlay_hash_from_metadata=lambda meta: None,
            overlay_hash_label="registry_hash",
            skip_reserved_hash_validation="yes",  # type: ignore[arg-type]
        )
