"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_lifecycle_module_layout.py

Layout contract tests for USR lifecycle command registration decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli as usr_cli


def test_usr_cli_lifecycle_registration_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.lifecycle")
    assert hasattr(module, "register_lifecycle_commands")
    assert hasattr(module, "SnapshotDeps")
    assert hasattr(module, "cmd_snapshot")

    cli_module = importlib.import_module("dnadesign.usr.src.cli_commands.lifecycle.cli")
    assert hasattr(cli_module, "register_lifecycle_commands")

    snapshot_module = importlib.import_module("dnadesign.usr.src.cli_commands.lifecycle.snapshot")
    assert hasattr(snapshot_module, "SnapshotDeps")
    assert hasattr(snapshot_module, "cmd_snapshot")


def test_usr_cli_uses_lifecycle_command_registrar() -> None:
    source = inspect.getsource(usr_cli)
    assert "register_lifecycle_commands(" in source
