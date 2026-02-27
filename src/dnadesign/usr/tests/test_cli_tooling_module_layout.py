"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_tooling_module_layout.py

Layout contract tests for USR tooling command decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli as usr_cli


def test_usr_cli_tooling_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.tooling")
    assert hasattr(module, "ToolingDeps")
    assert hasattr(module, "cmd_repair_densegen")
    assert hasattr(module, "cmd_convert_legacy")
    assert hasattr(module, "cmd_make_mock")
    assert hasattr(module, "cmd_add_demo")


def test_usr_cli_tooling_commands_delegate_to_module() -> None:
    source = inspect.getsource(usr_cli)
    assert "tooling_commands.cmd_repair_densegen(" in source
    assert "tooling_commands.cmd_convert_legacy(" in source
    assert "tooling_commands.cmd_make_mock(" in source
    assert "tooling_commands.cmd_add_demo(" in source
