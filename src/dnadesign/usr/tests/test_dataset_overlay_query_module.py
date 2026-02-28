"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_dataset_overlay_query_module.py

Layout contract tests for Dataset overlay-query decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.dataset import Dataset


def test_dataset_overlay_query_module_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.dataset_overlay_query")


def test_dataset_overlay_query_method_delegates_to_module() -> None:
    source = inspect.getsource(Dataset._duckdb_query)
    assert "build_overlay_query(" in source
