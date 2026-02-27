"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_sync_execution_module_layout.py

Layout contract tests for sync execution orchestration decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.cli_commands import sync as sync_commands


def test_sync_execution_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.sync_execution")
    assert hasattr(module, "SyncExecutionDeps")
    assert hasattr(module, "assert_dataset_only_flags_for_file_mode")
    assert hasattr(module, "run_file_sync")
    assert hasattr(module, "resolve_pull_dataset_target")
    assert hasattr(module, "resolve_push_dataset_target")
    assert hasattr(module, "run_dataset_sync")


def test_sync_module_delegates_execution_helpers() -> None:
    source = inspect.getsource(sync_commands)
    assert "sync_execution_commands.run_file_sync(" in source
    assert "sync_execution_commands.resolve_pull_dataset_target(" in source
    assert "sync_execution_commands.resolve_push_dataset_target(" in source
    assert "sync_execution_commands.run_dataset_sync(" in source
