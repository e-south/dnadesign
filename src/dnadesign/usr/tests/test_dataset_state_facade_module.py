"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_dataset_state_facade_module.py

Layout contract tests for Dataset state-facade decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.dataset import Dataset


def test_dataset_state_facade_module_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.dataset_state_facade")


def test_dataset_state_methods_delegate_to_facade_module() -> None:
    ensure_source = inspect.getsource(Dataset._ensure_ids_exist)
    tombstone_source = inspect.getsource(Dataset.tombstone)
    restore_source = inspect.getsource(Dataset.restore)
    set_state_source = inspect.getsource(Dataset.set_state)
    clear_state_source = inspect.getsource(Dataset.clear_state)
    get_state_source = inspect.getsource(Dataset.get_state)

    assert "ensure_dataset_ids_exist(" in ensure_source
    assert "tombstone_dataset_rows(" in tombstone_source
    assert "restore_dataset_rows(" in restore_source
    assert "set_dataset_state_fields(" in set_state_source
    assert "clear_dataset_state_fields(" in clear_state_source
    assert "get_dataset_state_frame(" in get_state_source
