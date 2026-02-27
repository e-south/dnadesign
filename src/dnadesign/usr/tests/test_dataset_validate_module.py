"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_dataset_validate_module.py

Layout contract tests for Dataset validation decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.dataset import Dataset


def test_dataset_validate_module_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.dataset_validate")


def test_dataset_validate_method_delegates_to_validate_module() -> None:
    source = inspect.getsource(Dataset.validate)
    assert "validate_dataset(" in source
