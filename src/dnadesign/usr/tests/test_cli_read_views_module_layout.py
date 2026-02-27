"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_read_views_module_layout.py

Layout contract tests for USR read view command decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli as usr_cli


def test_usr_cli_read_views_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.read_views")
    assert hasattr(module, "cmd_head")
    assert hasattr(module, "cmd_cols")
    assert hasattr(module, "cmd_describe")
    assert hasattr(module, "cmd_cell")


def test_usr_cli_read_view_commands_delegate_to_read_views_module() -> None:
    source = inspect.getsource(usr_cli)
    assert "read_views_commands.cmd_head(" in source
    assert "read_views_commands.cmd_cols(" in source
    assert "read_views_commands.cmd_describe(" in source
    assert "read_views_commands.cmd_cell(" in source
