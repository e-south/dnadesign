"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_dataset_overlay_catalog_module.py

Layout contract tests for Dataset overlay catalog decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.dataset import Dataset


def test_dataset_overlay_catalog_module_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.dataset_overlay_catalog")


def test_dataset_overlay_catalog_methods_delegate_to_module() -> None:
    load_source = inspect.getsource(Dataset._load_overlays)
    info_source = inspect.getsource(Dataset.info)
    schema_source = inspect.getsource(Dataset.schema)

    assert "load_overlay_catalog(" in load_source
    assert "build_dataset_info(" in info_source
    assert "merge_dataset_schema(" in schema_source
