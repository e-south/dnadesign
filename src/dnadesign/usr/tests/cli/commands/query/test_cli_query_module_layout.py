"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/cli/commands/query/test_cli_query_module_layout.py

Layout contract tests for USR query command registration decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli.support.wiring.registration as cli_registration_support


def test_usr_cli_query_package_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.query")
    assert hasattr(module, "register_query_commands")


def test_usr_cli_query_registration_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.query.cli")
    assert hasattr(module, "register_query_commands")


def test_usr_cli_uses_query_command_registrar() -> None:
    source = inspect.getsource(cli_registration_support.register_cli_surface)
    assert "register_query_commands(" in source
