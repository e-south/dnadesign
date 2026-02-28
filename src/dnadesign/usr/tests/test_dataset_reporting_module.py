"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_dataset_reporting_module.py

Layout contract tests for Dataset reporting/profile decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.dataset import Dataset


def test_dataset_reporting_module_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.dataset_reporting")


def test_dataset_manifest_and_describe_delegate_to_reporting_module() -> None:
    manifest_source = inspect.getsource(Dataset.manifest)
    manifest_dict_source = inspect.getsource(Dataset.manifest_dict)
    describe_source = inspect.getsource(Dataset.describe)

    assert "manifest_dataset(" in manifest_source
    assert "manifest_dict_dataset(" in manifest_dict_source
    assert "describe_dataset(" in describe_source
