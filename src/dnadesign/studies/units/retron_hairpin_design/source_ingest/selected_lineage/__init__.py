"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/selected_lineage/__init__.py

Public selected-lineage contract and loading surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .contracts import (
    MaterializedVariantLineageEntryV1,
    MaterializedVariantLineageError,
    MaterializedVariantLineageV1,
    MsdStructuralPrimitiveRefsV1,
)
from .loading import load_lineage_document
from .validation import validate_lineage


def load_materialized_variant_lineage(
    path: str | Path,
    *,
    repo_root: str | Path,
) -> MaterializedVariantLineageV1:
    """Load and verify selected lineage against owner-controlled source artifacts."""

    lineage, root = load_lineage_document(path, repo_root=repo_root)
    validate_lineage(lineage, repo_root=root)
    return lineage


__all__ = [
    "MaterializedVariantLineageEntryV1",
    "MaterializedVariantLineageError",
    "MaterializedVariantLineageV1",
    "MsdStructuralPrimitiveRefsV1",
    "load_materialized_variant_lineage",
]
