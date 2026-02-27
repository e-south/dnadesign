"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_dataset_views_module.py

Layout contract tests for Dataset read/export view decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.dataset import Dataset


def test_dataset_views_module_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.dataset_views")


def test_dataset_read_and_export_methods_delegate_to_views_module() -> None:
    scan_source = inspect.getsource(Dataset.scan)
    head_source = inspect.getsource(Dataset.head)
    get_source = inspect.getsource(Dataset.get)
    grep_source = inspect.getsource(Dataset.grep)
    export_source = inspect.getsource(Dataset.export)

    assert "scan_dataset(" in scan_source
    assert "head_dataset(" in head_source
    assert "get_dataset(" in get_source
    assert "grep_dataset(" in grep_source
    assert "export_dataset(" in export_source
