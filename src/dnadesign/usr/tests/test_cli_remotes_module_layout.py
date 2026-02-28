"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_remotes_module_layout.py

Layout contract tests for USR remotes command decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli as usr_cli


def test_usr_cli_remotes_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.remotes")
    assert hasattr(module, "cmd_remotes_list")
    assert hasattr(module, "cmd_remotes_show")
    assert hasattr(module, "cmd_remotes_add")
    assert hasattr(module, "cmd_remotes_wizard")
    assert hasattr(module, "cmd_remotes_doctor")


def test_usr_cli_remotes_commands_delegate_to_remotes_module() -> None:
    source = inspect.getsource(usr_cli)
    assert "remotes_commands.cmd_remotes_list(" in source
    assert "remotes_commands.cmd_remotes_show(" in source
    assert "remotes_commands.cmd_remotes_add(" in source
    assert "remotes_commands.cmd_remotes_wizard(" in source
    assert "remotes_commands.cmd_remotes_doctor(" in source


def test_usr_cli_uses_remotes_command_registrar() -> None:
    source = inspect.getsource(usr_cli)
    assert "register_remotes_commands(" in source
