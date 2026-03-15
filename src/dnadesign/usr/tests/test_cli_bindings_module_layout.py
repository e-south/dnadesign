"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_bindings_module_layout.py

Layout contract tests for CLI command binding helpers extracted from cli.py.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli_bindings as cli_bindings


def test_usr_cli_bindings_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_bindings")
    assert hasattr(module, "CliBindings")
    assert hasattr(module, "build_cli_bindings")


def test_usr_cli_bindings_delegate_to_command_modules() -> None:
    source = inspect.getsource(cli_bindings)
    assert "read_views_commands.cmd_cell(" in source
    assert "runtime_commands.cmd_validate(" in source
    assert "maintenance_commands.cmd_overlay_remove(" in source
    assert "namespace_handlers_commands.cmd_namespace_register(" in source
    assert "merge_commands.cmd_merge_datasets(" in source
    assert "tooling_commands.cmd_repair_densegen(" in source
