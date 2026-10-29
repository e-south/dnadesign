"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/tests/datasets/core/test_dataset_identity_module.py

Layout contract tests for dataset identity helpers extracted from Dataset.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.seqledger.src.dataset import Dataset


def test_dataset_identity_module_importable() -> None:
    module = importlib.import_module("dnadesign.seqledger.src.datasets.core.identity")
    assert hasattr(module, "normalize_dataset_id")
    assert hasattr(module, "open_dataset")


def test_dataset_open_delegates_to_identity_module() -> None:
    source = inspect.getsource(Dataset.open)
    assert "open_dataset(" in source
