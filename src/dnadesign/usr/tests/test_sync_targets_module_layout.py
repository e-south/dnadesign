"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_sync_targets_module_layout.py

Layout contract tests for sync target/path helper decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

from dnadesign.usr.src.cli_commands import sync as sync_commands


def test_sync_targets_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.sync_targets")
    assert hasattr(module, "is_file_mode_target")
    assert hasattr(module, "is_dataset_dir_target")
    assert hasattr(module, "resolve_dataset_dir_target")
    assert hasattr(module, "resolve_remote_path_for_file")
    assert hasattr(module, "resolve_dataset_id_for_diff_or_pull")


def test_sync_module_delegates_target_helpers() -> None:
    source = inspect.getsource(sync_commands)
    assert "sync_targets_commands.is_file_mode_target(" in source
    assert "sync_targets_commands.is_dataset_dir_target(" in source
    assert "sync_targets_commands.resolve_dataset_dir_target(" in source
    assert "sync_targets_commands.resolve_remote_path_for_file(" in source
    assert "sync_targets_commands.resolve_dataset_id_for_diff_or_pull(" in source
