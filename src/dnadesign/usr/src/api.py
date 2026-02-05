"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/api.py

Public API surface for USR (library-first).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .dataset import Dataset
from .errors import (
    AlphabetError,
    DuplicateIDError,
    EmbeddingDimensionError,
    NamespaceError,
    SchemaError,
    SequencesError,
    ValidationError,
)
from .normalize import compute_id, normalize_sequence, validate_alphabet, validate_bio_type
from .schema import ARROW_SCHEMA, ID_HASH_SPEC, REQUIRED_COLUMNS, SCHEMA_VERSION
from .types import AddSequencesResult, DatasetInfo, Fingerprint, Manifest, OverlayInfo
from .version import __version__

__all__ = [
    "Dataset",
    "AlphabetError",
    "DuplicateIDError",
    "EmbeddingDimensionError",
    "NamespaceError",
    "SchemaError",
    "SequencesError",
    "ValidationError",
    "compute_id",
    "normalize_sequence",
    "validate_bio_type",
    "validate_alphabet",
    "ARROW_SCHEMA",
    "REQUIRED_COLUMNS",
    "SCHEMA_VERSION",
    "ID_HASH_SPEC",
    "Fingerprint",
    "OverlayInfo",
    "Manifest",
    "DatasetInfo",
    "AddSequencesResult",
    "__version__",
]
