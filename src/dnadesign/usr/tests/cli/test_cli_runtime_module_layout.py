"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_runtime_module_layout.py

Layout contract tests for USR runtime command decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli_support.bindings as cli_bindings


def test_usr_cli_runtime_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.query")
    assert hasattr(module, "RuntimeDeps")
    assert hasattr(module, "cmd_validate")
    assert hasattr(module, "cmd_events_tail")
    assert hasattr(module, "cmd_get")
    assert hasattr(module, "cmd_grep")
    assert hasattr(module, "cmd_export")


def test_usr_cli_runtime_commands_delegate_to_runtime_module() -> None:
    source = inspect.getsource(cli_bindings)
    assert "query_commands.cmd_validate(" in source
    assert "query_commands.cmd_events_tail(" in source
    assert "query_commands.cmd_get(" in source
    assert "query_commands.cmd_grep(" in source
    assert "query_commands.cmd_export(" in source
