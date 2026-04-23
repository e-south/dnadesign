"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/source_layout_inventory.py

Single source of truth for the sanctioned USR source layout.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

TOP_LEVEL_SOURCE_MODULES = {
    "__init__.py",
    "api.py",
    "cli.py",
    "convert_legacy.py",
    "dataset.py",
    "duckdb_runtime.py",
    "errors.py",
    "events.py",
    "maintenance.py",
    "normalize.py",
    "overlays.py",
    "registry.py",
    "schema.py",
    "sync.py",
    "types.py",
    "version.py",
}

TOP_LEVEL_SOURCE_PACKAGES = {
    "cli_commands",
    "cli_support",
    "datasets",
    "legacy",
    "overlay_support",
    "remote_sync",
    "storage",
}

HELPER_PACKAGE_FILES = {
    "cli_commands": {
        "__init__.py",
        "deps.py",
        "error_output.py",
    },
    "cli_support": {
        "__init__.py",
        "bindings.py",
        "event_output.py",
        "merge_policy.py",
        "paths.py",
        "pretty.py",
        "rendering.py",
        "roots.py",
        "stderr_filter.py",
        "surface.py",
    },
    "datasets": {
        "__init__.py",
        "activity.py",
        "dedupe.py",
        "events.py",
        "identity.py",
        "ingest.py",
        "materialize.py",
        "mock.py",
        "reserved_overlay.py",
    },
    "legacy": {
        "__init__.py",
        "dedupe.py",
        "inputs.py",
        "tfbs.py",
    },
    "overlay_support": {
        "__init__.py",
        "digest_ledger.py",
        "maintenance.py",
        "projection.py",
    },
    "remote_sync": {
        "__init__.py",
        "config.py",
        "diff.py",
        "execution.py",
        "remote.py",
        "sidecars.py",
        "transfer.py",
    },
    "storage": {
        "__init__.py",
        "locking.py",
        "parquet.py",
    },
}

HELPER_PACKAGE_SUBPACKAGES = {
    "cli_commands": {
        "datasets",
        "lifecycle",
        "maintenance",
        "namespace",
        "query",
        "read_views",
        "remotes",
        "sync",
        "tooling",
    },
    "cli_support": set(),
    "datasets": {"lifecycle", "merge", "overlay", "query", "state", "validate", "views"},
    "legacy": set(),
    "overlay_support": set(),
    "remote_sync": set(),
    "storage": set(),
}

NESTED_PACKAGE_FILES = {
    ("cli_commands", "datasets"): {
        "__init__.py",
        "catalog.py",
        "resolution.py",
    },
    ("cli_commands", "lifecycle"): {
        "__init__.py",
        "cli.py",
        "materialize.py",
        "snapshot.py",
        "state.py",
        "write.py",
    },
    ("cli_commands", "maintenance"): {
        "__init__.py",
        "cli.py",
        "dedupe.py",
        "merge.py",
        "overlay.py",
        "registry.py",
    },
    ("cli_commands", "sync"): {
        "__init__.py",
        "cli.py",
        "execution.py",
        "output.py",
        "policy.py",
        "targets.py",
    },
    ("cli_commands", "read_views"): {
        "__init__.py",
        "parquet_targets.py",
    },
    ("cli_commands", "namespace"): {
        "__init__.py",
        "cli.py",
    },
    ("cli_commands", "query"): {
        "__init__.py",
        "cli.py",
        "read.py",
        "runtime.py",
    },
    ("cli_commands", "remotes"): {
        "__init__.py",
        "cli.py",
    },
    ("cli_commands", "tooling"): {
        "__init__.py",
        "cli.py",
        "densegen.py",
        "dev.py",
        "legacy.py",
        "shared.py",
    },
    ("datasets", "lifecycle"): {
        "__init__.py",
        "registry.py",
        "write_session.py",
    },
    ("datasets", "merge"): {
        "__init__.py",
        "execution.py",
        "overlay_carry.py",
    },
    ("datasets", "query"): {
        "__init__.py",
        "catalog.py",
        "planner.py",
    },
    ("datasets", "overlay"): {
        "__init__.py",
        "attach.py",
        "maintenance.py",
        "policy.py",
        "write.py",
    },
    ("datasets", "state"): {
        "__init__.py",
        "facade.py",
    },
    ("datasets", "validate"): {
        "__init__.py",
        "registry_modes.py",
    },
    ("datasets", "views"): {
        "__init__.py",
        "read_keys.py",
        "reporting.py",
    },
}
