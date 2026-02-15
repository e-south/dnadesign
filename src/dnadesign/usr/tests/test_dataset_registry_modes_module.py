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


def test_normalize_registry_mode_rejects_unknown_mode() -> None:
    with pytest.raises(SchemaError, match="Unsupported registry_mode"):
        normalize_registry_mode("mystery")


def test_register_registry_mode_rejects_duplicate_name() -> None:
    mode_name = "unit_custom_mode"
    register_registry_mode(
        mode=mode_name,
        allowed_hashes=lambda dataset: set(),
        validate_with_registries=lambda dataset, validate: None,
    )
    with pytest.raises(SchemaError, match=f"registry_mode '{mode_name}' is already registered"):
        register_registry_mode(
            mode=mode_name,
            allowed_hashes=lambda dataset: set(),
            validate_with_registries=lambda dataset, validate: None,
        )
