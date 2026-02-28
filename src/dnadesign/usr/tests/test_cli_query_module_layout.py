"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_query_module_layout.py

Layout contract tests for USR query command registration decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli as usr_cli


def test_usr_cli_query_registration_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.query_cli")
    assert hasattr(module, "register_query_commands")


def test_usr_cli_uses_query_command_registrar() -> None:
    source = inspect.getsource(usr_cli)
    assert "register_query_commands(" in source
