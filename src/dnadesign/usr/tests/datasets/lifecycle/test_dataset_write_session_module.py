"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/datasets/lifecycle/test_dataset_write_session_module.py

Layout contract tests for Dataset write-session decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.dataset import Dataset


def test_dataset_write_session_module_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.datasets.lifecycle.write_session")


def test_dataset_write_session_methods_delegate_to_session_module() -> None:
    init_source = inspect.getsource(Dataset.init)
    write_session_source = inspect.getsource(Dataset.write_session)

    assert "dataset_lifecycle.init_dataset(" in init_source
    assert "dataset_lifecycle.DatasetWriteSession(self)" in write_session_source
