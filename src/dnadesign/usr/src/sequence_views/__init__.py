"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/sequence_views/__init__.py

Public sequence-view helpers for USR semantic product aliases.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .maintenance import SequenceViewAliasRepairResult, repair_sequence_view_alias_conflicts
from .models import (
    SEQUENCE_VIEW_SIDECAR_RELATIVE_PATH,
    VIEW_ID_SCHEMA_VERSION,
    VIEW_SEMANTICS_SIDECAR_RELATIVE_PATH,
    ContextKind,
    Orientation,
    PoolingOperation,
    ProductKind,
    SequenceViewConflictPolicy,
    SequenceViewRecord,
    SequenceViewSelector,
    SequenceViewSemanticKey,
    ViewSemanticsConflictPolicy,
    ViewSemanticsRecord,
    compute_sequence_view_id,
)
from .qa import (
    SequenceViewContractExpectation,
    SequenceViewContractReport,
    validate_sequence_view_contract,
)
from .semantics import (
    load_view_semantics,
    load_view_semantics_index,
    view_semantics_path,
    write_view_semantics,
)
from .store import (
    load_sequence_view_ids,
    load_sequence_view_index,
    load_sequence_views,
    select_sequence_views,
    sequence_views_path,
    write_sequence_views,
)

__all__ = [
    "SEQUENCE_VIEW_SIDECAR_RELATIVE_PATH",
    "VIEW_SEMANTICS_SIDECAR_RELATIVE_PATH",
    "ContextKind",
    "Orientation",
    "PoolingOperation",
    "ProductKind",
    "SequenceViewConflictPolicy",
    "SequenceViewRecord",
    "SequenceViewSemanticKey",
    "SequenceViewSelector",
    "SequenceViewContractExpectation",
    "SequenceViewContractReport",
    "SequenceViewAliasRepairResult",
    "ViewSemanticsConflictPolicy",
    "ViewSemanticsRecord",
    "VIEW_ID_SCHEMA_VERSION",
    "compute_sequence_view_id",
    "load_sequence_view_index",
    "load_sequence_views",
    "load_sequence_view_ids",
    "load_view_semantics",
    "load_view_semantics_index",
    "repair_sequence_view_alias_conflicts",
    "select_sequence_views",
    "validate_sequence_view_contract",
    "view_semantics_path",
    "sequence_views_path",
    "write_view_semantics",
    "write_sequence_views",
]
