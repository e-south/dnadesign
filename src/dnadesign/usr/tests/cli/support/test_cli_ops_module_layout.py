"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/cli/support/test_cli_ops_module_layout.py

Layout contract tests for USR operations command registration decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli.support.wiring.registration as cli_registration_support


def test_usr_cli_ops_registration_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.tooling")
    assert hasattr(module, "register_ops_commands")


def test_usr_cli_tooling_registration_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.tooling.cli")
    assert hasattr(module, "register_ops_commands")


def test_usr_cli_uses_ops_command_registrar() -> None:
    source = inspect.getsource(cli_registration_support.register_cli_surface)
    assert "register_ops_commands(" in source
