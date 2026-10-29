"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/tests/datasets/merge/test_dataset_merge_package_module.py

Layout contract tests for Dataset merge package decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib


def test_dataset_merge_package_importable() -> None:
    assert importlib.import_module("dnadesign.seqledger.src.datasets.merge")
    assert importlib.import_module("dnadesign.seqledger.src.datasets.merge.execution")
    assert importlib.import_module("dnadesign.seqledger.src.datasets.merge.overlay_carry")
