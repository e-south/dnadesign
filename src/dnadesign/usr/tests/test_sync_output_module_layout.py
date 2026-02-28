"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_sync_output_module_layout.py

Layout contract tests for sync output rendering decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.cli_commands import sync as sync_commands


def test_sync_output_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.sync_output")
    assert hasattr(module, "build_sync_audit_payload")
    assert hasattr(module, "print_diff")
    assert hasattr(module, "print_verify_notes")
    assert hasattr(module, "print_sync_audit")


def test_sync_module_delegates_output_rendering() -> None:
    source = inspect.getsource(sync_commands)
    assert "sync_output_commands.print_diff(" in source
    assert "sync_output_commands.print_verify_notes(" in source
    assert "sync_output_commands.build_sync_audit_payload(" in source
    assert "sync_output_commands.print_sync_audit(" in source
