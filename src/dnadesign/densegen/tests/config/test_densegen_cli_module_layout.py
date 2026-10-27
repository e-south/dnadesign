"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/config/test_densegen_cli_module_layout.py

Tests that CLI helper modules are organized under cli/ package.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.util


def test_top_level_cli_helper_modules_are_not_present() -> None:
    assert importlib.util.find_spec("dnadesign.densegen.src.cli_render") is None
    assert importlib.util.find_spec("dnadesign.densegen.src.cli_sampling") is None
    assert importlib.util.find_spec("dnadesign.densegen.src.cli_setup") is None
    cli_commands_spec = importlib.util.find_spec("dnadesign.densegen.src.cli_commands")
    assert cli_commands_spec is None or cli_commands_spec.loader is None
    assert importlib.util.find_spec("dnadesign.densegen.src.cli") is not None
    assert importlib.util.find_spec("dnadesign.densegen.src.cli.main") is not None
