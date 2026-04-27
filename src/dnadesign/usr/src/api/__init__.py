"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/api/__init__.py

Public API surface for USR (library-first).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..cli.support.resolution.roots import default_usr_root, normalize_usr_root
from ..contracts import (
    ARROW_SCHEMA,
    ID_HASH_SPEC,
    REQUIRED_COLUMNS,
    SCHEMA_VERSION,
    AddSequencesResult,
    AlphabetError,
    DatasetInfo,
    DuplicateIDError,
    EmbeddingDimensionError,
    Fingerprint,
    Manifest,
    NamespaceError,
    OverlayInfo,
    SchemaError,
    SequencesError,
    ValidationError,
    compute_id,
    normalize_sequence,
    validate_alphabet,
    validate_bio_type,
)
from ..dataset import Dataset
from ..events import USR_EVENT_VERSION
from ..registry import (
    DERIVED_NAMESPACE,
    SEQ_ANNOT_NAMESPACE,
    USR_LABEL_NAMESPACE,
    ensure_registry_entries,
    ensure_sequence_contract_namespaces,
)
from ..sequence_views import (
    SEQUENCE_VIEW_SIDECAR_RELATIVE_PATH,
    VIEW_ID_SCHEMA_VERSION,
    ContextKind,
    Orientation,
    PoolingOperation,
    ProductKind,
    SequenceViewConflictPolicy,
    SequenceViewRecord,
    SequenceViewSelector,
    SequenceViewSemanticKey,
    compute_sequence_view_id,
    load_sequence_views,
    select_sequence_views,
    sequence_views_path,
    write_sequence_views,
)
from ..version import __version__

__all__ = [
    "Dataset",
    "USR_EVENT_VERSION",
    "AlphabetError",
    "DuplicateIDError",
    "EmbeddingDimensionError",
    "NamespaceError",
    "SchemaError",
    "SequencesError",
    "ValidationError",
    "compute_id",
    "default_usr_root",
    "normalize_sequence",
    "normalize_usr_root",
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
    "USR_LABEL_NAMESPACE",
    "SEQ_ANNOT_NAMESPACE",
    "DERIVED_NAMESPACE",
    "ProductKind",
    "Orientation",
    "ContextKind",
    "PoolingOperation",
    "SequenceViewConflictPolicy",
    "SequenceViewSemanticKey",
    "SequenceViewRecord",
    "SequenceViewSelector",
    "VIEW_ID_SCHEMA_VERSION",
    "SEQUENCE_VIEW_SIDECAR_RELATIVE_PATH",
    "compute_sequence_view_id",
    "sequence_views_path",
    "load_sequence_views",
    "select_sequence_views",
    "write_sequence_views",
    "ensure_registry_entries",
    "ensure_sequence_contract_namespaces",
    "__version__",
]
