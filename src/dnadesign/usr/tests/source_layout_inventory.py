"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/source_layout_inventory.py

Single source of truth for the sanctioned USR source layout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

TOP_LEVEL_SOURCE_MODULES = {
    "__init__.py",
}

TOP_LEVEL_SOURCE_PACKAGES = {
    "api",
    "cli",
    "contracts",
    "dataset",
    "datasets",
    "events",
    "genbank",
    "legacy",
    "maintenance",
    "overlays",
    "regulondb",
    "registry",
    "runtime",
    "sequence_views",
    "storage",
    "sync",
    "version",
}

PACKAGE_FILES = {
    ("api",): {
        "__init__.py",
    },
    ("cli",): {
        "__init__.py",
    },
    ("cli", "commands"): {
        "__init__.py",
        "deps.py",
        "error_output.py",
    },
    ("cli", "support"): {
        "__init__.py",
    },
    ("contracts",): {
        "__init__.py",
        "errors.py",
        "normalize.py",
        "schema.py",
        "types.py",
    },
    ("dataset",): {
        "__init__.py",
    },
    ("datasets",): {
        "__init__.py",
    },
    ("datasets", "core"): {
        "__init__.py",
        "activity.py",
        "events.py",
        "identity.py",
        "ingest.py",
    },
    ("datasets", "demo"): {
        "__init__.py",
        "mock.py",
    },
    ("events",): {
        "__init__.py",
        "actor.py",
        "append.py",
        "defaults.py",
        "fingerprint.py",
        "gardening.py",
        "recording.py",
        "redaction.py",
    },
    ("genbank",): {
        "__init__.py",
        "importer.py",
        "models.py",
        "parser.py",
    },
    ("legacy",): {
        "__init__.py",
        "convert.py",
        "dedupe.py",
        "inputs.py",
        "tfbs.py",
    },
    ("maintenance",): {
        "__init__.py",
    },
    ("overlays",): {
        "__init__.py",
        "constants.py",
        "metadata.py",
        "paths.py",
    },
    ("overlays", "support"): {
        "__init__.py",
        "digest_ledger.py",
        "maintenance.py",
        "projection.py",
    },
    ("regulondb",): {
        "__init__.py",
        "functional_annotations.py",
    },
    ("registry",): {
        "__init__.py",
        "models.py",
        "storage.py",
        "typespec.py",
        "validation.py",
    },
    ("runtime",): {
        "__init__.py",
        "duckdb.py",
    },
    ("sequence_views",): {
        "__init__.py",
        "maintenance.py",
        "models.py",
        "qa.py",
        "semantics.py",
        "store.py",
    },
    ("storage",): {
        "__init__.py",
        "locking.py",
        "parquet.py",
    },
    ("sync",): {
        "__init__.py",
    },
    ("sync", "remote"): {
        "__init__.py",
        "config.py",
        "diff.py",
        "execution.py",
        "locks.py",
        "remote.py",
        "sidecars.py",
        "transfer.py",
    },
    ("version",): {
        "__init__.py",
    },
    ("cli", "commands", "datasets"): {
        "__init__.py",
        "catalog.py",
        "resolution.py",
    },
    ("cli", "commands", "genbank"): {
        "__init__.py",
        "cli.py",
    },
    ("cli", "commands", "lifecycle"): {
        "__init__.py",
        "cli.py",
        "materialize.py",
        "snapshot.py",
        "state.py",
        "write.py",
    },
    ("cli", "commands", "maintenance"): {
        "__init__.py",
        "cli.py",
        "dedupe.py",
        "events.py",
        "merge.py",
        "overlay.py",
        "registry.py",
    },
    ("cli", "commands", "namespace"): {
        "__init__.py",
        "cli.py",
    },
    ("cli", "commands", "query"): {
        "__init__.py",
        "cli.py",
        "read.py",
        "runtime.py",
    },
    ("cli", "commands", "read_views"): {
        "__init__.py",
        "parquet_targets.py",
    },
    ("cli", "commands", "remotes"): {
        "__init__.py",
        "cli.py",
    },
    ("cli", "commands", "sync"): {
        "__init__.py",
        "cli.py",
        "execution.py",
        "output.py",
        "policy.py",
        "targets.py",
    },
    ("cli", "commands", "tooling"): {
        "__init__.py",
        "cli.py",
        "densegen.py",
        "dev.py",
        "legacy.py",
        "shared.py",
    },
    ("cli", "support", "presentation"): {
        "__init__.py",
        "event_output.py",
        "pretty.py",
        "rendering.py",
        "runtime.py",
        "stderr_filter.py",
    },
    ("cli", "support", "resolution"): {
        "__init__.py",
        "dataset_targets.py",
        "merge_policy.py",
        "paths.py",
        "roots.py",
    },
    ("cli", "support", "wiring"): {
        "__init__.py",
        "bindings.py",
        "dependencies.py",
        "registration.py",
        "surface.py",
    },
    ("datasets", "lifecycle"): {
        "__init__.py",
        "materialize.py",
        "registry.py",
        "snapshot.py",
        "write_session.py",
    },
    ("datasets", "maintenance"): {
        "__init__.py",
        "dedupe.py",
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
        "reserved_overlay.py",
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

PACKAGE_SUBPACKAGES = {
    ("api",): set(),
    ("cli",): {
        "commands",
        "support",
    },
    ("cli", "commands"): {
        "datasets",
        "genbank",
        "lifecycle",
        "maintenance",
        "namespace",
        "query",
        "read_views",
        "remotes",
        "sync",
        "tooling",
    },
    ("cli", "commands", "datasets"): set(),
    ("cli", "commands", "genbank"): set(),
    ("cli", "commands", "lifecycle"): set(),
    ("cli", "commands", "maintenance"): set(),
    ("cli", "commands", "namespace"): set(),
    ("cli", "commands", "query"): set(),
    ("cli", "commands", "read_views"): set(),
    ("cli", "commands", "remotes"): set(),
    ("cli", "commands", "sync"): set(),
    ("cli", "commands", "tooling"): set(),
    ("cli", "support"): {"presentation", "resolution", "wiring"},
    ("cli", "support", "presentation"): set(),
    ("cli", "support", "resolution"): set(),
    ("cli", "support", "wiring"): set(),
    ("contracts",): set(),
    ("dataset",): set(),
    ("datasets",): {
        "core",
        "demo",
        "lifecycle",
        "maintenance",
        "merge",
        "overlay",
        "query",
        "state",
        "validate",
        "views",
    },
    ("datasets", "core"): set(),
    ("datasets", "demo"): set(),
    ("datasets", "lifecycle"): set(),
    ("datasets", "maintenance"): set(),
    ("datasets", "merge"): set(),
    ("datasets", "overlay"): set(),
    ("datasets", "query"): set(),
    ("datasets", "state"): set(),
    ("datasets", "validate"): set(),
    ("datasets", "views"): set(),
    ("events",): set(),
    ("genbank",): set(),
    ("legacy",): set(),
    ("maintenance",): set(),
    ("overlays",): {"support"},
    ("overlays", "support"): set(),
    ("regulondb",): set(),
    ("registry",): set(),
    ("runtime",): set(),
    ("sequence_views",): set(),
    ("storage",): set(),
    ("sync",): {"remote"},
    ("sync", "remote"): set(),
    ("version",): set(),
}
