"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_dataset_materialize_module.py

Layout contract tests for Dataset materialize decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.dataset import Dataset


def test_dataset_materialize_module_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.dataset_materialize")


def test_dataset_materialize_delegates_to_module_function() -> None:
    source = inspect.getsource(Dataset.materialize)
    assert "materialize_dataset(" in source
