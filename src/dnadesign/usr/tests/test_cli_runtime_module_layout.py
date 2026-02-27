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

import dnadesign.usr.src.cli as usr_cli


def test_usr_cli_runtime_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.runtime")
    assert hasattr(module, "RuntimeDeps")
    assert hasattr(module, "cmd_validate")
    assert hasattr(module, "cmd_events_tail")
    assert hasattr(module, "cmd_get")
    assert hasattr(module, "cmd_grep")
    assert hasattr(module, "cmd_export")
    assert hasattr(module, "cmd_delete")
    assert hasattr(module, "cmd_restore")
    assert hasattr(module, "cmd_state_set")
    assert hasattr(module, "cmd_state_clear")
    assert hasattr(module, "cmd_state_get")


def test_usr_cli_runtime_commands_delegate_to_runtime_module() -> None:
    source = inspect.getsource(usr_cli)
    assert "runtime_commands.cmd_validate(" in source
    assert "runtime_commands.cmd_events_tail(" in source
    assert "runtime_commands.cmd_get(" in source
    assert "runtime_commands.cmd_grep(" in source
    assert "runtime_commands.cmd_export(" in source
    assert "runtime_commands.cmd_delete(" in source
    assert "runtime_commands.cmd_restore(" in source
    assert "runtime_commands.cmd_state_set(" in source
    assert "runtime_commands.cmd_state_clear(" in source
    assert "runtime_commands.cmd_state_get(" in source
