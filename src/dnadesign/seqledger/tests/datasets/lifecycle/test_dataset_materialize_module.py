"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/tests/datasets/lifecycle/test_dataset_materialize_module.py

Layout contract tests for Dataset materialize decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.seqledger.src.dataset import Dataset


def test_dataset_materialize_module_importable() -> None:
    assert importlib.import_module("dnadesign.seqledger.src.datasets.lifecycle.materialize")


def test_dataset_materialize_delegates_to_module_function() -> None:
    source = inspect.getsource(Dataset.materialize)
    assert "materialize_dataset(" in source
