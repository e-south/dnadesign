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


def test_usr_cli_maintenance_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.maintenance")
    assert hasattr(module, "MaintenanceDeps")
    assert hasattr(module, "cmd_registry_freeze")
    assert hasattr(module, "cmd_overlay_compact")
    assert hasattr(module, "cmd_snapshot")
    assert hasattr(module, "cmd_dedupe_sequences")


def test_usr_cli_maintenance_commands_delegate_to_maintenance_module() -> None:
    source = inspect.getsource(usr_cli)
    assert "maintenance_commands.cmd_registry_freeze(" in source
    assert "maintenance_commands.cmd_overlay_compact(" in source
    assert "maintenance_commands.cmd_snapshot(" in source
    assert "maintenance_commands.cmd_dedupe_sequences(" in source
