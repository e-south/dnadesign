"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_dataset_overlay_maintenance_module.py

Layout contract tests for Dataset overlay maintenance decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.dataset import Dataset


def test_dataset_overlay_maintenance_module_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.dataset_overlay_maintenance")


def test_dataset_overlay_maintenance_methods_delegate_to_module() -> None:
    list_source = inspect.getsource(Dataset.list_overlays)
    remove_source = inspect.getsource(Dataset.remove_overlay)
    compact_source = inspect.getsource(Dataset.compact_overlay)

    assert "list_overlay_infos(" in list_source
    assert "remove_overlay_namespace(" in remove_source
    assert "compact_overlay_namespace(" in compact_source
