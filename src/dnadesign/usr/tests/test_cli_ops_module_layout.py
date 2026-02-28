"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_ops_module_layout.py

Layout contract tests for USR operations command registration decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli as usr_cli


def test_usr_cli_ops_registration_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.ops_cli")
    assert hasattr(module, "register_ops_commands")


def test_usr_cli_uses_ops_command_registrar() -> None:
    source = inspect.getsource(usr_cli)
    assert "register_ops_commands(" in source
