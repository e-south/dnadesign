"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/datasets/lifecycle/test_dataset_lifecycle_package_module.py

Layout contract tests for Dataset lifecycle package decomposition.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.dataset import Dataset


def test_dataset_lifecycle_package_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.datasets.lifecycle")
    assert importlib.import_module("dnadesign.usr.src.datasets.lifecycle.registry")
    assert importlib.import_module("dnadesign.usr.src.datasets.lifecycle.snapshot")


def test_dataset_lifecycle_methods_delegate_to_lifecycle_package() -> None:
    freeze_source = inspect.getsource(Dataset.freeze_registry)
    snapshot_source = inspect.getsource(Dataset.snapshot)
    require_exists_source = inspect.getsource(Dataset._require_exists)
    require_registry_source = inspect.getsource(Dataset._require_registry_for_mutation)
    metadata_source = inspect.getsource(Dataset._base_metadata)
    tombstone_source = inspect.getsource(Dataset._tombstone_path)
    registry_source = inspect.getsource(Dataset._registry)
    registry_hash_source = inspect.getsource(Dataset._registry_hash)
    stored_hash_source = inspect.getsource(Dataset._dataset_registry_hash)
    frozen_path_source = inspect.getsource(Dataset._frozen_registry_path)
    auto_freeze_source = inspect.getsource(Dataset._auto_freeze_registry)

    assert "dataset_lifecycle.freeze_registry(self)" in freeze_source
    assert "dataset_lifecycle.snapshot_dataset(self)" in snapshot_source
    assert "dataset_lifecycle.require_dataset_exists(" in require_exists_source
    assert "dataset_lifecycle.require_registry_for_mutation(" in require_registry_source
    assert "dataset_lifecycle.base_metadata(" in metadata_source
    assert "dataset_lifecycle.tombstone_path(" in tombstone_source
    assert "dataset_lifecycle.load_dataset_registry(" in registry_source
    assert "dataset_lifecycle.dataset_registry_hash(" in registry_hash_source
    assert "dataset_lifecycle.stored_dataset_registry_hash(" in stored_hash_source
    assert "dataset_lifecycle.frozen_registry_path(" in frozen_path_source
    assert "dataset_lifecycle.auto_freeze_registry(" in auto_freeze_source
