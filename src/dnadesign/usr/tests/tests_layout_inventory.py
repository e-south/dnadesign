"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/tests_layout_inventory.py

Single source of truth for the sanctioned USR test layout.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

ROOT_TEST_SUPPORT_FILES = {
    "__init__.py",
    "source_layout_inventory.py",
    "tests_layout_inventory.py",
}

ROOT_TEST_MODULES = {
    "test_module_layout.py",
    "test_public_api_imports.py",
    "test_root_resolution_contract.py",
    "test_tests_layout.py",
    "test_usr_harness_script.py",
    "test_usr_sync_audit_drill_script.py",
}

TOP_LEVEL_TEST_PACKAGES = {
    "cli",
    "docs_contract",
    "datasets",
    "legacy",
    "overlays",
    "sync",
}

TEST_FAMILY_FILES = {
    "cli": {
        "__init__.py",
        "test_cli_command_surface.py",
        "test_cli_imports.py",
        "test_cli_root_contract.py",
        "test_cli_strict.py",
        "test_cli_typer.py",
    },
    "docs_contract": {
        "__init__.py",
        "helpers.py",
        "test_layout.py",
        "test_navigation.py",
        "test_study.py",
        "test_sync.py",
    },
    "datasets": {
        "__init__.py",
    },
    "legacy": {
        "__init__.py",
        "test_convert_module_imports.py",
        "test_legacy_dedupe_module.py",
        "test_legacy_inputs_module.py",
        "test_legacy_tfbs_module.py",
    },
    "overlays": {
        "__init__.py",
        "test_overlays.py",
    },
    "sync": {
        "__init__.py",
        "test_sync_iterative_batch_flow.py",
        "test_sync_locking.py",
        "test_sync_module_layout.py",
        "test_sync_remote_failures.py",
        "test_sync_schema_adversarial.py",
    },
}

TEST_FAMILY_SUBPACKAGES = {
    "cli": {"commands", "support", "sync"},
    "docs_contract": set(),
    "datasets": {"core", "lifecycle", "merge", "overlay", "query", "state", "validate", "views"},
    "legacy": set(),
    "overlays": {"support"},
    "sync": {"remote"},
}

NESTED_TEST_PACKAGE_SUBPACKAGES = {
    ("cli", "commands"): {
        "datasets",
        "lifecycle",
        "maintenance",
        "namespace",
        "query",
        "read_views",
        "remotes",
        "tooling",
    },
}

NESTED_TEST_PACKAGE_FILES = {
    ("cli", "commands"): {
        "__init__.py",
    },
    ("cli", "commands", "datasets"): {
        "__init__.py",
        "test_cli_archived_paths.py",
        "test_cli_datasets_package_module.py",
    },
    ("cli", "commands", "lifecycle"): {
        "__init__.py",
        "test_cli_lifecycle_module_layout.py",
        "test_cli_materialize.py",
        "test_cli_materialize_module_layout.py",
        "test_cli_materialize_prompt.py",
        "test_cli_snapshot.py",
        "test_cli_state.py",
    },
    ("cli", "commands", "maintenance"): {
        "__init__.py",
        "test_cli_maintenance_module_layout.py",
        "test_cli_maintenance_registry.py",
        "test_cli_merge_module_layout.py",
    },
    ("cli", "commands", "namespace"): {
        "__init__.py",
        "test_cli_namespace_handlers_module_layout.py",
        "test_cli_namespace_module_layout.py",
    },
    ("cli", "commands", "query"): {
        "__init__.py",
        "test_cli_events_tail.py",
        "test_cli_export.py",
        "test_cli_format_json.py",
        "test_cli_get.py",
        "test_cli_query_module_layout.py",
    },
    ("cli", "commands", "read_views"): {
        "__init__.py",
        "test_cli_read_parquet_targets_module.py",
        "test_cli_read_views_module_layout.py",
        "test_read_views_head_order.py",
    },
    ("cli", "commands", "remotes"): {
        "__init__.py",
        "test_cli_remotes_module_layout.py",
        "test_cli_remotes_wizard.py",
    },
    ("cli", "commands", "tooling"): {
        "__init__.py",
        "test_cli_tooling_module_layout.py",
    },
    ("cli", "support"): {
        "__init__.py",
        "test_cli_bindings_module_layout.py",
        "test_cli_deps_module_layout.py",
        "test_cli_error_output_module_layout.py",
        "test_cli_event_output_module.py",
        "test_cli_merge_policy_registry_module.py",
        "test_cli_ops_module_layout.py",
        "test_cli_runtime_module_layout.py",
        "test_stderr_filter.py",
        "test_ui_rich.py",
    },
    ("cli", "sync"): {
        "__init__.py",
        "test_cli_sync_args_builder_layout.py",
        "test_cli_sync_bootstrap_resolution.py",
        "test_cli_sync_execution_module_layout.py",
        "test_cli_sync_output_module_layout.py",
        "test_cli_sync_target_modes.py",
        "test_cli_sync_targets_module_layout.py",
    },
    ("overlays", "support"): {
        "__init__.py",
        "test_overlay_digest_ledger.py",
        "test_overlay_maintenance_module.py",
        "test_overlay_projection.py",
    },
    ("sync", "remote"): {
        "__init__.py",
        "test_diff_resilience.py",
        "test_remote_control_session.py",
        "test_remote_inventory_paths.py",
        "test_remote_lock_handshake.py",
        "test_remote_rsync_contract.py",
        "test_remote_transport_failures.py",
        "test_remotes_config.py",
        "test_verify_mode.py",
    },
    ("datasets", "core"): {
        "__init__.py",
        "test_dataset_activity_module.py",
        "test_dataset_dedupe_module.py",
        "test_dataset_events_module.py",
        "test_dataset_identity_module.py",
        "test_dataset_ingest_module.py",
        "test_dataset_layout.py",
        "test_dedupe.py",
        "test_duckdb_session_contract.py",
        "test_events_schema.py",
        "test_import_strict.py",
        "test_normalize_id.py",
    },
    ("datasets", "lifecycle"): {
        "__init__.py",
        "test_dataset_lifecycle_package_module.py",
        "test_dataset_materialize_module.py",
        "test_dataset_write_session.py",
        "test_dataset_write_session_module.py",
        "test_locking.py",
        "test_maintenance_context.py",
        "test_materialize_snapshot_streaming.py",
        "test_registry.py",
        "test_registry_autofreeze.py",
    },
    ("datasets", "merge"): {
        "__init__.py",
        "test_dataset_merge_package_module.py",
        "test_merge_locking.py",
        "test_merge_overlay_carry.py",
        "test_merge_streaming.py",
    },
    ("datasets", "overlay"): {
        "__init__.py",
        "test_attach_duckdb.py",
        "test_attach_sequence_keys.py",
        "test_attach_strict.py",
        "test_dataset_overlay_maintenance_module.py",
        "test_dataset_overlay_package_module.py",
        "test_dataset_reserved_overlay_module.py",
        "test_mock_overlay.py",
    },
    ("datasets", "query"): {
        "__init__.py",
        "test_dataset_overlay_catalog_module.py",
        "test_dataset_overlay_query_module.py",
    },
    ("datasets", "state"): {
        "__init__.py",
        "test_dataset_state_facade_module.py",
        "test_tombstones.py",
        "test_usr_state.py",
    },
    ("datasets", "validate"): {
        "__init__.py",
        "test_dataset_registry_modes_module.py",
        "test_dataset_validate_module.py",
        "test_validate_streaming.py",
    },
    ("datasets", "views"): {
        "__init__.py",
        "test_dataset_read_keys_module.py",
        "test_dataset_read_ops.py",
        "test_dataset_reporting_module.py",
        "test_dataset_scan_projection.py",
        "test_dataset_views_module.py",
    },
}
