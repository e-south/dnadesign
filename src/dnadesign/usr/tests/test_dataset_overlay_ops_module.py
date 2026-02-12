"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_dataset_overlay_ops_module.py

Layout contract tests for Dataset overlay attach/write decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.dataset import Dataset


def test_dataset_overlay_ops_module_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.dataset_overlay_ops")


def test_dataset_attach_and_write_overlay_delegate_to_overlay_ops_module() -> None:
    attach_source = inspect.getsource(Dataset.attach)
    write_overlay_source = inspect.getsource(Dataset.write_overlay)
    write_part_source = inspect.getsource(Dataset.write_overlay_part)

    assert "attach_dataset(" in attach_source
    assert "write_overlay_dataset(" in write_overlay_source
    assert "write_overlay_part_dataset(" in write_part_source
