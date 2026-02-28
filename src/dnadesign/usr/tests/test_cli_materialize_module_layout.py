"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_materialize_module_layout.py

Layout contract tests for USR materialize command decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli as usr_cli


def test_usr_cli_materialize_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.materialize")
    assert hasattr(module, "MaterializeDeps")
    assert hasattr(module, "cmd_materialize")


def test_usr_cli_materialize_command_delegates_to_materialize_module() -> None:
    source = inspect.getsource(usr_cli)
    assert "materialize_commands.cmd_materialize(" in source
