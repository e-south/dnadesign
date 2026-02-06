"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_module_layout.py

Module layout contract tests for USR package decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib


def test_storage_package_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.storage.parquet")
    assert hasattr(module, "write_parquet_atomic")
    assert hasattr(module, "iter_parquet_batches")


def test_cli_commands_dataset_helpers_available() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.datasets")
    assert hasattr(module, "list_datasets")
    assert hasattr(module, "resolve_existing_dataset_id")


def test_cli_paths_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_paths")
    assert hasattr(module, "assert_supported_root")
    assert hasattr(module, "resolve_dataset_for_read")
    assert hasattr(module, "resolve_path_anywhere")


def test_dataset_query_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.dataset_query")
    assert hasattr(module, "sql_ident")
    assert hasattr(module, "sql_str")
    assert hasattr(module, "create_overlay_view")


def test_cli_sync_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.sync")
    assert hasattr(module, "cmd_diff")
    assert hasattr(module, "cmd_pull")
    assert hasattr(module, "cmd_push")


def test_cli_read_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.read")
    assert hasattr(module, "cmd_ls")
    assert hasattr(module, "cmd_info")
    assert hasattr(module, "cmd_schema")


def test_cli_write_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.write")
    assert hasattr(module, "cmd_init")
    assert hasattr(module, "cmd_import")
    assert hasattr(module, "cmd_attach")


def test_cli_state_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.state")
    assert hasattr(module, "cmd_delete")
    assert hasattr(module, "cmd_restore")
    assert hasattr(module, "cmd_state_set")
    assert hasattr(module, "cmd_state_clear")
    assert hasattr(module, "cmd_state_get")


def test_dataset_state_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.dataset_state")
    assert hasattr(module, "ensure_ids_exist")
    assert hasattr(module, "tombstone")
    assert hasattr(module, "restore")
    assert hasattr(module, "set_state")
    assert hasattr(module, "clear_state")
    assert hasattr(module, "get_state")
