"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/contracts/__init__.py

Shared USR contract surfaces for errors, schema, API types, and sequence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module

_ATTR_SOURCES: dict[str, tuple[str, str]] = {
    "AlphabetError": (".errors", "AlphabetError"),
    "DuplicateGroup": (".errors", "DuplicateGroup"),
    "DuplicateIDError": (".errors", "DuplicateIDError"),
    "EmbeddingDimensionError": (".errors", "EmbeddingDimensionError"),
    "NamespaceError": (".errors", "NamespaceError"),
    "RemoteConfigError": (".errors", "RemoteConfigError"),
    "RemoteUnavailableError": (".errors", "RemoteUnavailableError"),
    "SchemaError": (".errors", "SchemaError"),
    "SequencesError": (".errors", "SequencesError"),
    "TransferError": (".errors", "TransferError"),
    "UserAbort": (".errors", "UserAbort"),
    "ValidationError": (".errors", "ValidationError"),
    "VerificationError": (".errors", "VerificationError"),
    "ALLOWED_BIO_TYPES": (".normalize", "ALLOWED_BIO_TYPES"),
    "ALPHABET_PATTERNS": (".normalize", "ALPHABET_PATTERNS"),
    "ALPHABETS_BY_BIO_TYPE": (".normalize", "ALPHABETS_BY_BIO_TYPE"),
    "ALPHABET_SYMBOLS": (".normalize", "ALPHABET_SYMBOLS"),
    "ID_DELIMITER": (".normalize", "ID_DELIMITER"),
    "compute_id": (".normalize", "compute_id"),
    "normalize_sequence": (".normalize", "normalize_sequence"),
    "validate_alphabet": (".normalize", "validate_alphabet"),
    "validate_bio_type": (".normalize", "validate_bio_type"),
    "ARROW_SCHEMA": (".schema", "ARROW_SCHEMA"),
    "ID_HASH_SPEC": (".schema", "ID_HASH_SPEC"),
    "META_DATASET_CREATED_AT": (".schema", "META_DATASET_CREATED_AT"),
    "META_ID_HASH": (".schema", "META_ID_HASH"),
    "META_REGISTRY_HASH": (".schema", "META_REGISTRY_HASH"),
    "META_SCHEMA_VERSION": (".schema", "META_SCHEMA_VERSION"),
    "REQUIRED_COLUMNS": (".schema", "REQUIRED_COLUMNS"),
    "SCHEMA_VERSION": (".schema", "SCHEMA_VERSION"),
    "base_metadata": (".schema", "base_metadata"),
    "merge_base_metadata": (".schema", "merge_base_metadata"),
    "with_base_metadata": (".schema", "with_base_metadata"),
    "AddSequencesResult": (".types", "AddSequencesResult"),
    "DatasetInfo": (".types", "DatasetInfo"),
    "Fingerprint": (".types", "Fingerprint"),
    "Manifest": (".types", "Manifest"),
    "OverlayInfo": (".types", "OverlayInfo"),
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
