"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/config/test_public_api_module_layout.py

Tests that DenseGen exposes its public API from package root without top-level
facade module sprawl.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.util

import dnadesign.densegen as densegen


def test_densegen_public_api_is_root_scoped() -> None:
    assert callable(densegen.app)
    assert callable(densegen.load_config)
    assert callable(densegen.resolve_run_root)
    assert callable(densegen.resolve_outputs_scoped_path)
    assert callable(densegen.densegen_notebook_render_contract)
    assert densegen.PLOT_SPECS


def test_densegen_top_level_facade_modules_are_not_present() -> None:
    assert importlib.util.find_spec("dnadesign.densegen.cli") is None
    assert importlib.util.find_spec("dnadesign.densegen.config") is None
    assert importlib.util.find_spec("dnadesign.densegen.notebook_contract") is None
    assert importlib.util.find_spec("dnadesign.densegen.plot_registry") is None
    assert importlib.util.find_spec("dnadesign.densegen.__main__") is not None
