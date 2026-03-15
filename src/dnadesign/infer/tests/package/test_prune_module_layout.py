"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/package/test_prune_module_layout.py

Layout contract tests for infer prune delegation into USR overlay maintenance.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect

from dnadesign.infer.src.prune import prune_usr_overlay


def test_prune_usr_overlay_delegates_to_usr_overlay_maintenance() -> None:
    source = inspect.getsource(prune_usr_overlay)
    assert "remove_dataset_overlay(" in source
