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


def test_dataset_reporting_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.dataset_reporting")
    assert hasattr(module, "manifest_dataset")
    assert hasattr(module, "manifest_dict_dataset")
    assert hasattr(module, "describe_dataset")


def test_dataset_views_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.dataset_views")
    assert hasattr(module, "scan_dataset")
    assert hasattr(module, "head_dataset")
    assert hasattr(module, "get_dataset")
    assert hasattr(module, "grep_dataset")
    assert hasattr(module, "export_dataset")


def test_dataset_ingest_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.dataset_ingest")
    assert hasattr(module, "prepare_import_rows_dataset")
    assert hasattr(module, "write_import_df_dataset")
    assert hasattr(module, "import_rows_dataset")
    assert hasattr(module, "add_sequences_dataset")
    assert hasattr(module, "import_csv_dataset")
    assert hasattr(module, "import_jsonl_dataset")


def test_dataset_validate_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.dataset_validate")
    assert hasattr(module, "validate_dataset")


def test_dataset_overlay_maintenance_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.dataset_overlay_maintenance")
    assert hasattr(module, "list_overlay_infos")
    assert hasattr(module, "remove_overlay_namespace")
    assert hasattr(module, "compact_overlay_namespace")


def test_dataset_reserved_overlay_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.dataset_reserved_overlay")
    assert hasattr(module, "write_reserved_overlay")


def test_dataset_overlay_query_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.dataset_overlay_query")
    assert hasattr(module, "build_overlay_query")


def test_cli_sync_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.sync")
    assert hasattr(module, "cmd_diff")
    assert hasattr(module, "cmd_pull")
    assert hasattr(module, "cmd_push")


def test_cli_sync_output_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.sync_output")
    assert hasattr(module, "build_sync_audit_payload")
    assert hasattr(module, "print_diff")
    assert hasattr(module, "print_verify_notes")
    assert hasattr(module, "print_sync_audit")


def test_cli_sync_targets_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.sync_targets")
    assert hasattr(module, "is_file_mode_target")
    assert hasattr(module, "is_dataset_dir_target")
    assert hasattr(module, "resolve_dataset_dir_target")
    assert hasattr(module, "resolve_remote_path_for_file")
    assert hasattr(module, "resolve_dataset_id_for_diff_or_pull")


def test_cli_sync_execution_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.sync_execution")
    assert hasattr(module, "SyncExecutionDeps")
    assert hasattr(module, "SyncRunResult")
    assert hasattr(module, "assert_dataset_only_flags_for_file_mode")
    assert hasattr(module, "run_file_sync")
    assert hasattr(module, "resolve_pull_dataset_target")
    assert hasattr(module, "resolve_push_dataset_target")
    assert hasattr(module, "run_dataset_sync")


def test_cli_sync_registration_module_exports_register_function() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.sync_cli")
    assert hasattr(module, "register_sync_commands")


def test_cli_remotes_registration_module_exports_register_function() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.remotes_cli")
    assert hasattr(module, "register_remotes_commands")


def test_cli_namespace_registration_module_exports_register_function() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.namespace_cli")
    assert hasattr(module, "register_namespace_commands")


def test_cli_query_registration_module_exports_register_function() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.query_cli")
    assert hasattr(module, "register_query_commands")


def test_cli_lifecycle_registration_module_exports_register_function() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.lifecycle_cli")
    assert hasattr(module, "register_lifecycle_commands")


def test_cli_ops_registration_module_exports_register_function() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.ops_cli")
    assert hasattr(module, "register_ops_commands")


def test_cli_surface_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_surface")
    assert hasattr(module, "build_cli_apps")


def test_cli_read_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.read")
    assert hasattr(module, "cmd_ls")
    assert hasattr(module, "cmd_info")
    assert hasattr(module, "cmd_schema")


def test_cli_read_views_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.read_views")
    assert hasattr(module, "cmd_head")
    assert hasattr(module, "cmd_cols")
    assert hasattr(module, "cmd_describe")
    assert hasattr(module, "cmd_cell")


def test_cli_runtime_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.runtime")
    assert hasattr(module, "cmd_validate")
    assert hasattr(module, "cmd_events_tail")
    assert hasattr(module, "cmd_get")
    assert hasattr(module, "cmd_grep")
    assert hasattr(module, "cmd_export")
    assert hasattr(module, "cmd_delete")
    assert hasattr(module, "cmd_restore")
    assert hasattr(module, "cmd_state_set")
    assert hasattr(module, "cmd_state_clear")
    assert hasattr(module, "cmd_state_get")


def test_cli_materialize_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.materialize")
    assert hasattr(module, "cmd_materialize")


def test_cli_maintenance_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.maintenance")
    assert hasattr(module, "cmd_registry_freeze")
    assert hasattr(module, "cmd_overlay_compact")
    assert hasattr(module, "cmd_snapshot")
    assert hasattr(module, "cmd_dedupe_sequences")


def test_cli_merge_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.merge")
    assert hasattr(module, "cmd_merge_datasets")


def test_cli_namespace_handlers_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.namespace_handlers")
    assert hasattr(module, "cmd_namespace_list")
    assert hasattr(module, "cmd_namespace_show")
    assert hasattr(module, "cmd_namespace_register")


def test_cli_tooling_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.tooling")
    assert hasattr(module, "cmd_repair_densegen")
    assert hasattr(module, "cmd_convert_legacy")
    assert hasattr(module, "cmd_make_mock")
    assert hasattr(module, "cmd_add_demo")


def test_cli_error_output_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.error_output")
    assert hasattr(module, "print_user_error")


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


def test_sync_transfer_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.sync_transfer")
    assert hasattr(module, "make_pull_staging_dir")
    assert hasattr(module, "copy_file_atomic")
    assert hasattr(module, "collect_staged_entries")
    assert hasattr(module, "promote_staged_pull")
