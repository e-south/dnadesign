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
from pathlib import Path

from .source_layout_inventory import (
    PACKAGE_FILES,
    PACKAGE_SUBPACKAGES,
    TOP_LEVEL_SOURCE_MODULES,
    TOP_LEVEL_SOURCE_PACKAGES,
)

PUBLIC_FACADE_MODULES = {
    "__init__.py",
    "__main__.py",
}


def test_public_package_root_only_contains_intentional_facade_modules() -> None:
    package_root = Path(__file__).resolve().parents[1]
    actual = {path.name for path in package_root.glob("*.py")}
    assert actual == PUBLIC_FACADE_MODULES


def test_public_package_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr")
    assert hasattr(module, "Dataset")
    assert hasattr(module, "SEQ_ANNOT_NAMESPACE")
    assert hasattr(module, "DERIVED_NAMESPACE")
    assert hasattr(module, "SequenceViewRecord")
    assert hasattr(module, "write_sequence_views")
    assert hasattr(module, "load_sequence_view_index")
    assert hasattr(module, "load_sequence_view_ids")
    assert hasattr(module, "ViewSemanticsRecord")
    assert hasattr(module, "write_view_semantics")
    assert hasattr(module, "load_view_semantics_index")
    assert hasattr(module, "SequenceViewContractExpectation")
    assert hasattr(module, "validate_sequence_view_contract")
    assert hasattr(module, "RESERVED_NAMESPACES")
    assert hasattr(module, "MUTATION_RESERVED_NAMESPACES")
    assert hasattr(module, "load_overlay_catalog")
    assert hasattr(module, "build_dataset_info")
    assert hasattr(module, "merge_dataset_schema")
    assert hasattr(module, "attach_frame_dataset")
    assert hasattr(module, "write_overlay_dataset")
    assert hasattr(module, "write_overlay_part_dataset")
    assert hasattr(module, "overlay_metadata")
    assert hasattr(module, "overlay_parts")
    assert hasattr(module, "overlay_schema")
    assert hasattr(module, "OVERLAY_DIGEST_LEDGER_FILENAME")
    assert hasattr(module, "overlay_digest_ledger_path")
    assert hasattr(module, "default_usr_root")
    assert hasattr(module, "normalize_usr_root")
    assert hasattr(module, "parse_columns_spec")
    assert hasattr(module, "pkg_usr_root")
    assert hasattr(module, "register_namespace")
    assert hasattr(module, "resolve_usr_root_from_config")
    assert hasattr(module, "resolve_usr_root_from_env")
    assert hasattr(module, "app")
    assert hasattr(module, "main")


def test_storage_parquet_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.storage.parquet")
    assert hasattr(module, "commit_parquet_atomic_file")
    assert hasattr(module, "read_parquet_head")
    assert hasattr(module, "write_parquet_atomic")
    assert hasattr(module, "iter_parquet_batches")


def test_storage_locking_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.storage.locking")
    assert hasattr(module, "LOCK_FILENAME")
    assert hasattr(module, "dataset_write_lock")


def test_contracts_package_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.contracts")
    assert hasattr(module, "SchemaError")
    assert hasattr(module, "REQUIRED_COLUMNS")
    assert hasattr(module, "compute_id")
    assert hasattr(module, "Fingerprint")


def test_runtime_package_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.runtime")
    assert hasattr(module, "connect_duckdb_utc")


def test_genbank_package_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.genbank")
    assert hasattr(module, "BiopythonGenBankParser")
    assert hasattr(module, "GenBankImportManifest")
    assert hasattr(module, "import_genbank_manifest")


def test_api_package_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.api")
    assert hasattr(module, "Dataset")
    assert hasattr(module, "ensure_sequence_contract_namespaces")
    assert hasattr(module, "SequenceViewSemanticKey")
    assert hasattr(module, "load_sequence_view_index")
    assert hasattr(module, "load_sequence_view_ids")
    assert hasattr(module, "USR_EVENT_VERSION")
    assert hasattr(module, "__version__")


def test_cli_package_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli")
    assert hasattr(module, "app")
    assert hasattr(module, "main")
    assert hasattr(module, "cmd_cell")
    assert hasattr(module, "merge_usr_to_usr")


def test_legacy_convert_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.legacy.convert")
    assert hasattr(module, "ConvertStats")
    assert hasattr(module, "RepairStats")
    assert hasattr(module, "convert_legacy")
    assert hasattr(module, "repair_densegen_used_tfbs")


def test_dataset_package_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.dataset")
    assert hasattr(module, "Dataset")
    assert hasattr(module, "ARCHIVE_DATASET_PREFIX")
    assert hasattr(module, "RESERVED_NAMESPACES")
    assert hasattr(module, "MUTATION_RESERVED_NAMESPACES")


def test_events_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.events")
    assert hasattr(module, "USR_EVENT_VERSION")
    assert hasattr(module, "fingerprint_parquet")
    assert hasattr(module, "record_event")


def test_registry_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.registry")
    assert hasattr(module, "RegistryColumn")
    assert hasattr(module, "SEQ_ANNOT_NAMESPACE")
    assert hasattr(module, "ensure_sequence_contract_namespaces")
    assert hasattr(module, "load_registry")
    assert hasattr(module, "register_namespace")
    assert hasattr(module, "parse_columns_spec")
    assert hasattr(module, "registry_hash")


def test_sequence_views_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.sequence_views")
    assert hasattr(module, "SequenceViewRecord")
    assert hasattr(module, "SequenceViewSemanticKey")
    assert hasattr(module, "compute_sequence_view_id")
    assert hasattr(module, "write_sequence_views")
    assert hasattr(module, "load_sequence_view_index")
    assert hasattr(module, "load_sequence_view_ids")


def test_overlays_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.overlays")
    assert hasattr(module, "overlay_path")
    assert hasattr(module, "overlay_dir_path")
    assert hasattr(module, "overlay_parts")
    assert hasattr(module, "overlay_metadata")
    assert hasattr(module, "overlay_schema")
    assert hasattr(module, "with_overlay_metadata")


def test_maintenance_package_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.maintenance")
    assert hasattr(module, "MaintenanceContext")
    assert hasattr(module, "current_maintenance")
    assert hasattr(module, "maintenance")
    assert hasattr(module, "require_maintenance")


def test_sync_package_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.sync")
    assert hasattr(module, "SyncOptions")
    assert hasattr(module, "plan_diff")
    assert hasattr(module, "execute_pull")
    assert hasattr(module, "execute_push")


def test_version_package_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.version")
    assert hasattr(module, "__version__")


def test_usr_source_root_contains_only_sanctioned_modules_and_packages() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    actual_modules = {path.name for path in src_root.glob("*.py")}
    actual_packages = {path.name for path in src_root.iterdir() if path.is_dir() and (path / "__init__.py").exists()}

    assert actual_modules == TOP_LEVEL_SOURCE_MODULES
    assert actual_packages == TOP_LEVEL_SOURCE_PACKAGES


def test_usr_package_inventory_matches_layout_inventory() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"

    top_level_package_paths = {package_path for package_path in PACKAGE_FILES if len(package_path) == 1}
    assert {package_path[0] for package_path in top_level_package_paths} == TOP_LEVEL_SOURCE_PACKAGES
    assert set(PACKAGE_FILES) == set(PACKAGE_SUBPACKAGES)

    for package_path, expected_files in PACKAGE_FILES.items():
        package_root = src_root.joinpath(*package_path)
        actual_files = {path.name for path in package_root.glob("*.py")}
        actual_subpackages = {
            path.name for path in package_root.iterdir() if path.is_dir() and (path / "__init__.py").exists()
        }

        assert actual_files == expected_files
        assert actual_subpackages == PACKAGE_SUBPACKAGES[package_path]


def test_cli_commands_dataset_helpers_available() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.datasets")
    assert hasattr(module, "list_datasets")
    assert hasattr(module, "resolve_existing_dataset_id")
    assert hasattr(module, "resolve_dataset_name_interactive")


def test_cli_commands_dataset_catalog_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.datasets.catalog")
    assert hasattr(module, "list_datasets")


def test_cli_commands_dataset_resolution_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.datasets.resolution")
    assert hasattr(module, "resolve_existing_dataset_id")
    assert hasattr(module, "resolve_dataset_name_interactive")


def test_cli_deps_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.deps")
    assert hasattr(module, "build_read_view_deps")
    assert hasattr(module, "build_runtime_deps")
    assert hasattr(module, "build_materialize_deps")
    assert hasattr(module, "build_snapshot_deps")
    assert hasattr(module, "build_maintenance_deps")
    assert hasattr(module, "build_merge_deps")
    assert hasattr(module, "build_namespace_deps")
    assert hasattr(module, "build_tooling_deps")


def test_cli_paths_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.support.resolution.paths")
    assert hasattr(module, "assert_supported_root")
    assert hasattr(module, "resolve_dataset_for_read")
    assert hasattr(module, "resolve_path_anywhere")


def test_cli_bindings_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.support.wiring.bindings")
    assert hasattr(module, "CliBindings")
    assert hasattr(module, "build_cli_bindings")


def test_cli_dataset_targets_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.support.resolution.dataset_targets")
    assert hasattr(module, "normalize_dataset_id")
    assert hasattr(module, "resolve_existing_dataset_id")
    assert hasattr(module, "resolve_dataset_name_interactive")
    assert hasattr(module, "resolve_dataset_for_read")
    assert hasattr(module, "list_datasets")


def test_cli_dependency_support_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.support.wiring.dependencies")
    assert hasattr(module, "build_read_view_deps")
    assert hasattr(module, "build_runtime_deps")
    assert hasattr(module, "build_materialize_deps")
    assert hasattr(module, "build_snapshot_deps")
    assert hasattr(module, "build_maintenance_deps")
    assert hasattr(module, "build_merge_deps")
    assert hasattr(module, "build_namespace_deps")
    assert hasattr(module, "build_tooling_deps")


def test_cli_pretty_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.support.presentation.pretty")
    assert hasattr(module, "PrettyOpts")
    assert hasattr(module, "fmt_value")
    assert hasattr(module, "render_schema_tree")
    assert hasattr(module, "profile_table")
    assert hasattr(module, "profile_batches")


def test_cli_rendering_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.support.presentation.rendering")
    assert hasattr(module, "print_df_plain")
    assert hasattr(module, "render_table_rich")
    assert hasattr(module, "render_schema_tree_rich")
    assert hasattr(module, "render_diff_rich")


def test_cli_roots_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.support.resolution.roots")
    assert hasattr(module, "default_usr_root")
    assert hasattr(module, "normalize_usr_root")
    assert hasattr(module, "pkg_usr_root")
    assert hasattr(module, "resolve_usr_root_from_config")
    assert hasattr(module, "resolve_usr_root_from_env")


def test_cli_stderr_filter_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.support.presentation.stderr_filter")
    assert hasattr(module, "should_filter_pyarrow_sysctl")
    assert hasattr(module, "maybe_install_pyarrow_sysctl_filter")


def test_cli_registration_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.support.wiring.registration")
    assert hasattr(module, "build_root_callback")
    assert hasattr(module, "ctx_args")
    assert hasattr(module, "register_cli_surface")


def test_cli_runtime_support_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.support.presentation.runtime")
    assert hasattr(module, "resolve_output_format")
    assert hasattr(module, "print_json")
    assert hasattr(module, "is_interactive")


def test_sync_remote_config_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.sync.remote.config")
    assert hasattr(module, "SSHRemoteConfig")
    assert not hasattr(module, "default_config_path")
    assert hasattr(module, "locate_config")
    assert hasattr(module, "load_all")
    assert hasattr(module, "save_remote")
    assert hasattr(module, "get_remote")


def test_sync_remote_execution_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.sync.remote.execution")
    assert hasattr(module, "SyncRuntime")
    assert hasattr(module, "plan_diff")
    assert hasattr(module, "plan_diff_file")
    assert hasattr(module, "execute_pull")
    assert hasattr(module, "execute_pull_file")
    assert hasattr(module, "execute_push")
    assert hasattr(module, "execute_push_file")


def test_dataset_query_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.query")
    assert hasattr(module, "sql_ident")
    assert hasattr(module, "sql_str")
    assert hasattr(module, "create_overlay_view")
    assert hasattr(module, "build_overlay_query")
    assert hasattr(module, "load_overlay_catalog")
    assert hasattr(module, "build_dataset_info")
    assert hasattr(module, "merge_dataset_schema")


def test_dataset_lifecycle_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.lifecycle")
    assert hasattr(module, "DatasetWriteSession")
    assert hasattr(module, "init_dataset")
    assert hasattr(module, "freeze_registry")
    assert hasattr(module, "auto_freeze_registry")
    assert hasattr(module, "base_metadata")
    assert hasattr(module, "frozen_registry_path")
    assert hasattr(module, "snapshot_dataset")
    assert hasattr(module, "tombstone_path")


def test_dataset_merge_package_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.merge")
    assert hasattr(module, "MergeColumnsMode")
    assert hasattr(module, "MergePolicy")
    assert hasattr(module, "MergePreview")
    assert hasattr(module, "OverlayCarryPlan")
    assert hasattr(module, "apply_overlay_carry")
    assert hasattr(module, "merge_usr_to_usr")
    assert hasattr(module, "plan_overlay_carry")


def test_dataset_merge_execution_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.merge.execution")
    assert hasattr(module, "MergeColumnsMode")
    assert hasattr(module, "MergePolicy")
    assert hasattr(module, "MergePreview")
    assert hasattr(module, "merge_usr_to_usr")


def test_dataset_merge_overlay_carry_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.merge.overlay_carry")
    assert hasattr(module, "OverlayCarryPlan")
    assert hasattr(module, "apply_overlay_carry")
    assert hasattr(module, "plan_overlay_carry")


def test_dataset_reporting_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.views.reporting")
    assert hasattr(module, "manifest_dataset")
    assert hasattr(module, "manifest_dict_dataset")
    assert hasattr(module, "describe_dataset")


def test_dataset_views_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.views")
    assert hasattr(module, "scan_dataset")
    assert hasattr(module, "head_dataset")
    assert hasattr(module, "get_dataset")
    assert hasattr(module, "grep_dataset")
    assert hasattr(module, "export_dataset")


def test_dataset_ingest_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.core.ingest")
    assert hasattr(module, "prepare_import_rows_dataset")
    assert hasattr(module, "write_import_df_dataset")
    assert hasattr(module, "import_rows_dataset")
    assert hasattr(module, "add_sequences_dataset")
    assert hasattr(module, "import_csv_dataset")
    assert hasattr(module, "import_jsonl_dataset")


def test_dataset_mock_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.demo.mock")
    assert hasattr(module, "MockSpec")
    assert hasattr(module, "make_mock_tables")
    assert hasattr(module, "create_mock_dataset")
    assert hasattr(module, "add_demo_columns")


def test_dataset_validate_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.validate")
    assert hasattr(module, "validate_dataset")


def test_dataset_validate_registry_modes_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.validate.registry_modes")
    assert hasattr(module, "normalize_registry_mode")
    assert hasattr(module, "register_registry_mode")


def test_dataset_identity_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.core.identity")
    assert hasattr(module, "normalize_dataset_id")
    assert hasattr(module, "open_dataset")


def test_dataset_read_keys_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.views.read_keys")
    assert hasattr(module, "key_list_from_batch")


def test_dataset_overlay_package_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.overlay")
    assert hasattr(module, "attach_dataset")
    assert hasattr(module, "attach_columns_dataset")
    assert hasattr(module, "attach_frame_dataset")
    assert hasattr(module, "write_overlay_dataset")
    assert hasattr(module, "write_overlay_part_dataset")
    assert hasattr(module, "list_overlay_infos")
    assert hasattr(module, "remove_overlay_namespace")
    assert hasattr(module, "compact_overlay_namespace")


def test_dataset_overlay_maintenance_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.overlay.maintenance")
    assert hasattr(module, "list_overlay_infos")
    assert hasattr(module, "remove_overlay_namespace")
    assert hasattr(module, "compact_overlay_namespace")


def test_dataset_overlay_policy_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.overlay.policy")
    assert hasattr(module, "validate_overlay_target")
    assert hasattr(module, "coerce_null_overlay_columns_to_registry_schema")


def test_overlays_support_maintenance_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.overlays.support.maintenance")
    assert hasattr(module, "remove_dataset_overlay")


def test_dataset_reserved_overlay_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.state.reserved_overlay")
    assert hasattr(module, "write_reserved_overlay")


def test_dataset_overlay_query_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.query.planner")
    assert hasattr(module, "build_overlay_query")


def test_dataset_overlay_catalog_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.query.catalog")
    assert hasattr(module, "load_overlay_catalog")
    assert hasattr(module, "build_dataset_info")
    assert hasattr(module, "merge_dataset_schema")


def test_dataset_state_facade_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.state.facade")
    assert hasattr(module, "ensure_dataset_ids_exist")
    assert hasattr(module, "tombstone_dataset_rows")
    assert hasattr(module, "restore_dataset_rows")
    assert hasattr(module, "set_dataset_state_fields")
    assert hasattr(module, "clear_dataset_state_fields")
    assert hasattr(module, "get_dataset_state_frame")


def test_cli_sync_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.sync")
    assert hasattr(module, "cmd_diff")
    assert hasattr(module, "cmd_pull")
    assert hasattr(module, "cmd_push")


def test_cli_sync_output_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.sync.output")
    assert hasattr(module, "build_sync_audit_payload")
    assert hasattr(module, "print_diff")
    assert hasattr(module, "print_verify_notes")
    assert hasattr(module, "print_sync_audit")


def test_cli_sync_targets_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.sync.targets")
    assert hasattr(module, "is_file_mode_target")
    assert hasattr(module, "is_dataset_dir_target")
    assert hasattr(module, "resolve_dataset_dir_target")
    assert hasattr(module, "resolve_remote_path_for_file")
    assert hasattr(module, "resolve_dataset_id_for_diff_or_pull")


def test_cli_sync_execution_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.sync.execution")
    assert hasattr(module, "SyncExecutionDeps")
    assert hasattr(module, "SyncRunResult")
    assert hasattr(module, "assert_dataset_only_flags_for_file_mode")
    assert hasattr(module, "run_file_sync")
    assert hasattr(module, "resolve_pull_dataset_target")
    assert hasattr(module, "resolve_push_dataset_target")
    assert hasattr(module, "run_dataset_sync")


def test_cli_sync_registration_module_exports_register_function() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.sync.cli")
    assert hasattr(module, "register_sync_commands")


def test_cli_remotes_registration_module_exports_register_function() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.remotes.cli")
    assert hasattr(module, "register_remotes_commands")


def test_cli_namespace_registration_module_exports_register_function() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.namespace.cli")
    assert hasattr(module, "register_namespace_commands")


def test_cli_query_registration_module_exports_register_function() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.query.cli")
    assert hasattr(module, "register_query_commands")


def test_cli_lifecycle_registration_module_exports_register_function() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.lifecycle.cli")
    assert hasattr(module, "register_lifecycle_commands")


def test_cli_ops_registration_module_exports_register_function() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.tooling")
    assert hasattr(module, "register_ops_commands")


def test_cli_tooling_registration_module_exports_register_function() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.tooling.cli")
    assert hasattr(module, "register_ops_commands")


def test_cli_surface_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.support.wiring.surface")
    assert hasattr(module, "build_cli_apps")


def test_devtools_usr_test_support_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.devtools.tests.support.usr")
    assert hasattr(module, "ensure_registry")
    assert hasattr(module, "register_test_namespace")


def test_cli_query_package_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.query")
    assert hasattr(module, "register_query_commands")
    assert hasattr(module, "RuntimeDeps")
    assert hasattr(module, "cmd_ls")
    assert hasattr(module, "cmd_info")
    assert hasattr(module, "cmd_schema")
    assert hasattr(module, "cmd_validate")
    assert hasattr(module, "cmd_events_tail")
    assert hasattr(module, "cmd_get")
    assert hasattr(module, "cmd_grep")
    assert hasattr(module, "cmd_export")


def test_cli_query_read_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.query.read")
    assert hasattr(module, "cmd_ls")
    assert hasattr(module, "cmd_info")
    assert hasattr(module, "cmd_schema")


def test_cli_read_views_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.read_views")
    assert hasattr(module, "cmd_head")
    assert hasattr(module, "cmd_cols")
    assert hasattr(module, "cmd_describe")
    assert hasattr(module, "cmd_cell")


def test_cli_read_parquet_targets_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.read_views.parquet_targets")
    assert hasattr(module, "_list_parquet_candidates")
    assert hasattr(module, "_resolve_parquet_from_dir")
    assert hasattr(module, "_resolve_parquet_target")
    assert hasattr(module, "_select_parquet_target_interactive")


def test_cli_runtime_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.query.runtime")
    assert hasattr(module, "cmd_validate")
    assert hasattr(module, "cmd_events_tail")
    assert hasattr(module, "cmd_get")
    assert hasattr(module, "cmd_grep")
    assert hasattr(module, "cmd_export")


def test_cli_lifecycle_package_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.lifecycle")
    assert hasattr(module, "register_lifecycle_commands")
    assert hasattr(module, "MaterializeDeps")
    assert hasattr(module, "SnapshotDeps")
    assert hasattr(module, "cmd_init")
    assert hasattr(module, "cmd_import")
    assert hasattr(module, "cmd_attach")
    assert hasattr(module, "cmd_delete")
    assert hasattr(module, "cmd_restore")
    assert hasattr(module, "cmd_state_set")
    assert hasattr(module, "cmd_state_clear")
    assert hasattr(module, "cmd_state_get")
    assert hasattr(module, "cmd_materialize")
    assert hasattr(module, "cmd_snapshot")


def test_cli_lifecycle_materialize_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.lifecycle.materialize")
    assert hasattr(module, "MaterializeDeps")
    assert hasattr(module, "cmd_materialize")


def test_cli_lifecycle_snapshot_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.lifecycle.snapshot")
    assert hasattr(module, "SnapshotDeps")
    assert hasattr(module, "cmd_snapshot")


def test_cli_maintenance_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.maintenance")
    assert hasattr(module, "register_maintenance_commands")
    assert hasattr(module, "cmd_registry_freeze")
    assert hasattr(module, "cmd_overlay_compact")
    assert hasattr(module, "cmd_overlay_project")
    assert hasattr(module, "cmd_overlay_remove")
    assert hasattr(module, "cmd_dedupe_sequences")
    assert hasattr(module, "MergeDeps")
    assert hasattr(module, "cmd_merge_datasets")


def test_cli_merge_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.maintenance.merge")
    assert hasattr(module, "cmd_merge_datasets")


def test_cli_namespace_handlers_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.namespace")
    assert hasattr(module, "cmd_namespace_list")
    assert hasattr(module, "cmd_namespace_show")
    assert hasattr(module, "cmd_namespace_register")


def test_cli_tooling_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.tooling")
    assert hasattr(module, "ToolingDeps")
    assert hasattr(module, "register_ops_commands")
    assert hasattr(module, "cmd_repair_densegen")
    assert hasattr(module, "cmd_convert_legacy")
    assert hasattr(module, "cmd_make_mock")
    assert hasattr(module, "cmd_add_demo")


def test_cli_tooling_densegen_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.tooling.densegen")
    assert hasattr(module, "cmd_repair_densegen")


def test_cli_tooling_dev_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.tooling.dev")
    assert hasattr(module, "cmd_make_mock")
    assert hasattr(module, "cmd_add_demo")


def test_cli_tooling_legacy_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.tooling.legacy")
    assert hasattr(module, "cmd_convert_legacy")


def test_cli_error_output_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.error_output")
    assert hasattr(module, "print_user_error")


def test_cli_write_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.lifecycle.write")
    assert hasattr(module, "cmd_init")
    assert hasattr(module, "cmd_import")
    assert hasattr(module, "cmd_attach")


def test_cli_state_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli.commands.lifecycle.state")
    assert hasattr(module, "cmd_delete")
    assert hasattr(module, "cmd_restore")
    assert hasattr(module, "cmd_state_set")
    assert hasattr(module, "cmd_state_clear")
    assert hasattr(module, "cmd_state_get")


def test_dataset_state_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.datasets.state")
    assert hasattr(module, "ensure_ids_exist")
    assert hasattr(module, "tombstone")
    assert hasattr(module, "restore")
    assert hasattr(module, "set_state")
    assert hasattr(module, "clear_state")
    assert hasattr(module, "get_state")


def test_sync_remote_transfer_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.sync.remote.transfer")
    assert hasattr(module, "make_pull_staging_dir")
    assert hasattr(module, "copy_file_atomic")
    assert hasattr(module, "collect_staged_entries")
    assert hasattr(module, "promote_staged_pull")


def test_legacy_inputs_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.legacy.inputs")
    assert hasattr(module, "Profile")
    assert hasattr(module, "profile_60bp_dual_promoter")
    assert hasattr(module, "_coerce_logits")
    assert hasattr(module, "_tf_from_parts")
    assert hasattr(module, "_count_tf")
    assert hasattr(module, "_ensure_pt_list_of_dicts")
    assert hasattr(module, "_gather_pt_files")


def test_legacy_tfbs_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.legacy.tfbs")
    assert hasattr(module, "_parse_tfbs_parts")
    assert hasattr(module, "_scan_used_tfbs")
    assert hasattr(module, "_detect_promoter_forward")


def test_legacy_dedupe_module_exports_expected_symbols() -> None:
    module = importlib.import_module("dnadesign.usr.src.legacy.dedupe")
    assert hasattr(module, "apply_casefold_sequence_dedupe")
