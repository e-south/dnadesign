"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/contracts/promoter_metadata.py

Shared promoter metadata contracts for LatentDNA workspaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

REGULONDB_NATIVE_PROMOTER_METADATA_COLUMNS = (
    "regulondb__sigma_factor_set",
    "regulondb__sigma_factor_count",
    "regulondb__regulator_composition",
    "regulondb__box_pattern",
    "regulondb__confidence_level_set",
    "regulondb__metadata_completeness_class",
    "regulondb__source_strata_set",
    "regulondb__primary_promoter_name",
)

__all__ = ["REGULONDB_NATIVE_PROMOTER_METADATA_COLUMNS"]
