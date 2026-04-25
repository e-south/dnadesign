"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/datasets/core/test_dataset_dedupe_module.py

Layout contract tests for Dataset dedupe decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.dataset import Dataset


def test_dataset_dedupe_module_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.datasets.maintenance.dedupe")


def test_dataset_dedupe_method_delegates_to_dedupe_module() -> None:
    source = inspect.getsource(Dataset.dedupe)
    assert "dedupe_dataset(" in source
