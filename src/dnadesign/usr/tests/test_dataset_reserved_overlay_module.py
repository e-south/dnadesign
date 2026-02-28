"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_dataset_reserved_overlay_module.py

Layout contract tests for Dataset reserved-overlay decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.dataset import Dataset


def test_dataset_reserved_overlay_module_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.dataset_reserved_overlay")


def test_dataset_reserved_overlay_method_delegates_to_module() -> None:
    source = inspect.getsource(Dataset._write_reserved_overlay)
    assert "write_reserved_overlay(" in source
