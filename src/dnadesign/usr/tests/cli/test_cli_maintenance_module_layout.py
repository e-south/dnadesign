"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_maintenance_module_layout.py

Layout contract tests for USR maintenance command decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli as usr_cli
import dnadesign.usr.src.cli_support.bindings as cli_bindings


def test_usr_cli_maintenance_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.maintenance")
    assert hasattr(module, "register_maintenance_commands")
    assert hasattr(module, "MaintenanceDeps")
    assert hasattr(module, "MergeDeps")
    assert hasattr(module, "cmd_registry_freeze")
    assert hasattr(module, "cmd_overlay_compact")
    assert hasattr(module, "cmd_overlay_project")
    assert hasattr(module, "cmd_overlay_remove")
    assert hasattr(module, "cmd_dedupe_sequences")
    assert hasattr(module, "cmd_merge_datasets")

    cli_module = importlib.import_module("dnadesign.usr.src.cli_commands.maintenance.cli")
    assert hasattr(cli_module, "register_maintenance_commands")


def test_usr_cli_uses_maintenance_command_registrar() -> None:
    source = inspect.getsource(usr_cli)
    assert "register_maintenance_commands(" in source


def test_usr_cli_maintenance_commands_delegate_to_maintenance_module() -> None:
    source = inspect.getsource(cli_bindings)
    assert "maintenance_commands.cmd_registry_freeze(" in source
    assert "maintenance_commands.cmd_overlay_compact(" in source
    assert "maintenance_commands.cmd_overlay_project(" in source
    assert "maintenance_commands.cmd_overlay_remove(" in source
    assert "maintenance_commands.cmd_dedupe_sequences(" in source
    assert "maintenance_commands.cmd_merge_datasets(" in source
