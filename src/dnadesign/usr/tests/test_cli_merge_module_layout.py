"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_merge_module_layout.py

Layout contract tests for USR merge command decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli_bindings as cli_bindings


def test_usr_cli_merge_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.merge")
    assert hasattr(module, "MergeDeps")
    assert hasattr(module, "cmd_merge_datasets")


def test_usr_cli_merge_command_delegates_to_merge_module() -> None:
    source = inspect.getsource(cli_bindings)
    assert "merge_commands.cmd_merge_datasets(" in source
