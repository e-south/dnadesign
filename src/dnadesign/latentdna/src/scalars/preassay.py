"""Pre-assay scalar builder registry."""

from __future__ import annotations

from typing import Any

from ..io.parquet_io import write_table
from ..workspaces.loader import WorkspaceContext
from .common import BuiltScalarArtifact
from .preassay_common import ScalarTableBuilder
from .preassay_ordinal import _ordinal_axis_audit_table
from .preassay_reference import _reference_alignment_summary_table, _reference_to_centroid_similarity_table
from .preassay_selection import (
    _axis_centroid_distance_table,
    _candidate_decision_frontier_table,
    _candidate_x_selection_scorecard_table,
    _collection_strength_ordinal_audit_table,
    _context_pair_summary_table,
    _ordinal_ladder_rows_table,
)
from .preassay_summary import (
    _cohort_structure_summary_table,
    _context_robustness_summary_table,
    _design_structure_summary_table,
    _representation_health_summary_table,
)

_PREASSAY_BUILDERS: dict[str, ScalarTableBuilder] = {
    "representation_health_summary": _representation_health_summary_table,
    "design_structure_summary": _design_structure_summary_table,
    "cohort_structure_summary": _cohort_structure_summary_table,
    "ordinal_axis_audit": _ordinal_axis_audit_table,
    "context_robustness_summary": _context_robustness_summary_table,
    "context_pair_summary": _context_pair_summary_table,
    "reference_alignment_summary": _reference_alignment_summary_table,
    "reference_to_centroid_similarity": _reference_to_centroid_similarity_table,
    "collection_strength_ordinal_audit": _collection_strength_ordinal_audit_table,
    "candidate_decision_frontier": _candidate_decision_frontier_table,
    "candidate_x_selection_scorecard": _candidate_x_selection_scorecard_table,
    "ordinal_ladder_rows": _ordinal_ladder_rows_table,
    "axis_centroid_distance": _axis_centroid_distance_table,
}

PREASSAY_BUILDER_KINDS = frozenset(_PREASSAY_BUILDERS)


def build_preassay_scalar_artifact(
    context: WorkspaceContext,
    *,
    scalar_id: str,
    builder_kind: str,
    params: dict[str, Any],
) -> BuiltScalarArtifact | None:
    builder = _PREASSAY_BUILDERS.get(builder_kind)
    if builder is None:
        return None
    table, inputs, stats = builder(context, params)
    artifact_dir = context.output_root / "scalars" / scalar_id
    write_table(table, artifact_dir / "table.parquet")
    return BuiltScalarArtifact(
        artifact_dir=artifact_dir,
        rows=table.num_rows,
        columns=table.column_names,
        inputs=inputs,
        outputs=[],
        stats=stats,
    )
