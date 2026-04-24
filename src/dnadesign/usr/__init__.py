"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/__init__.py

Public USR package surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module

_ATTR_SOURCES: dict[str, tuple[str, str]] = {
    "Dataset": (".src.api", "Dataset"),
    "USR_EVENT_VERSION": (".src.api", "USR_EVENT_VERSION"),
    "AlphabetError": (".src.api", "AlphabetError"),
    "DuplicateIDError": (".src.api", "DuplicateIDError"),
    "EmbeddingDimensionError": (".src.api", "EmbeddingDimensionError"),
    "NamespaceError": (".src.api", "NamespaceError"),
    "SchemaError": (".src.api", "SchemaError"),
    "SequencesError": (".src.api", "SequencesError"),
    "ValidationError": (".src.api", "ValidationError"),
    "compute_id": (".src.api", "compute_id"),
    "default_usr_root": (".src.api", "default_usr_root"),
    "normalize_sequence": (".src.api", "normalize_sequence"),
    "normalize_usr_root": (".src.api", "normalize_usr_root"),
    "validate_bio_type": (".src.api", "validate_bio_type"),
    "validate_alphabet": (".src.api", "validate_alphabet"),
    "ARROW_SCHEMA": (".src.api", "ARROW_SCHEMA"),
    "REQUIRED_COLUMNS": (".src.api", "REQUIRED_COLUMNS"),
    "SCHEMA_VERSION": (".src.api", "SCHEMA_VERSION"),
    "ID_HASH_SPEC": (".src.api", "ID_HASH_SPEC"),
    "Fingerprint": (".src.api", "Fingerprint"),
    "OverlayInfo": (".src.api", "OverlayInfo"),
    "Manifest": (".src.api", "Manifest"),
    "DatasetInfo": (".src.api", "DatasetInfo"),
    "AddSequencesResult": (".src.api", "AddSequencesResult"),
    "__version__": (".src.api", "__version__"),
    "main": (".src.cli", "main"),
    "app": (".src.cli", "app"),
    "pkg_usr_root": (".src.cli.support.resolution.roots", "pkg_usr_root"),
    "resolve_usr_root_from_config": (".src.cli.support.resolution.roots", "resolve_usr_root_from_config"),
    "resolve_usr_root_from_env": (".src.cli.support.resolution.roots", "resolve_usr_root_from_env"),
    "RESERVED_NAMESPACES": (".src.dataset", "RESERVED_NAMESPACES"),
    "MUTATION_RESERVED_NAMESPACES": (".src.dataset", "MUTATION_RESERVED_NAMESPACES"),
    "load_overlay_catalog": (".src.datasets.query", "load_overlay_catalog"),
    "build_dataset_info": (".src.datasets.query", "build_dataset_info"),
    "merge_dataset_schema": (".src.datasets.query", "merge_dataset_schema"),
    "attach_frame_dataset": (".src.datasets.overlay", "attach_frame_dataset"),
    "write_overlay_dataset": (".src.datasets.overlay", "write_overlay_dataset"),
    "write_overlay_part_dataset": (".src.datasets.overlay", "write_overlay_part_dataset"),
    "overlay_metadata": (".src.overlays", "overlay_metadata"),
    "overlay_parts": (".src.overlays", "overlay_parts"),
    "overlay_schema": (".src.overlays", "overlay_schema"),
    "OVERLAY_DIGEST_LEDGER_FILENAME": (".src.overlays.support.digest_ledger", "OVERLAY_DIGEST_LEDGER_FILENAME"),
    "OVERLAY_DIGEST_LEDGER_SCHEMA_VERSION": (
        ".src.overlays.support.digest_ledger",
        "OVERLAY_DIGEST_LEDGER_SCHEMA_VERSION",
    ),
    "build_overlay_digest_ledger": (".src.overlays.support.digest_ledger", "build_overlay_digest_ledger"),
    "overlay_digest_ledger_path": (".src.overlays.support.digest_ledger", "overlay_digest_ledger_path"),
    "update_overlay_digest_ledger": (".src.overlays.support.digest_ledger", "update_overlay_digest_ledger"),
    "write_overlay_digest_ledger": (".src.overlays.support.digest_ledger", "write_overlay_digest_ledger"),
}

__all__ = list(_ATTR_SOURCES)


def __getattr__(name: str):
    try:
        module_name, attr_name = _ATTR_SOURCES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
