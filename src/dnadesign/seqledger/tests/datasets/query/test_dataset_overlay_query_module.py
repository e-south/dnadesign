"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/tests/datasets/query/test_dataset_overlay_query_module.py

Layout contract tests for Dataset overlay-query decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.seqledger.src.dataset import Dataset


def test_dataset_overlay_query_module_importable() -> None:
    assert importlib.import_module("dnadesign.seqledger.src.datasets.query.planner")


def test_dataset_overlay_query_method_delegates_to_module() -> None:
    source = inspect.getsource(Dataset._duckdb_query)
    assert "build_overlay_query(" in source
