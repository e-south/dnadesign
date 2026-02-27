"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_namespace_module_layout.py

Layout contract tests for USR namespace command registration decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli as usr_cli


def test_usr_cli_namespace_registration_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.namespace_cli")
    assert hasattr(module, "register_namespace_commands")


def test_usr_cli_uses_namespace_command_registrar() -> None:
    source = inspect.getsource(usr_cli)
    assert "register_namespace_commands(" in source
