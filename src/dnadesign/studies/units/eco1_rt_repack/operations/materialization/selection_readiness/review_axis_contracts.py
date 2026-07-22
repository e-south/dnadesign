"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/review_axis_contracts.py

Review-axis field contracts for Eco1 RT panel selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewAxisMetric:
    """Triage-table field and notebook-facing label for a review metric."""

    field: str
    label: str


NA_FACING_CHARGE_FIELD = "nucleic_acid_facing_charge_delta"
NA_FACING_CHEMISTRY_METRICS = (
    ReviewAxisMetric("nucleic_acid_facing_basic_gain_count", "Basic gained"),
    ReviewAxisMetric("nucleic_acid_facing_basic_loss_count", "Basic lost"),
    ReviewAxisMetric("nucleic_acid_facing_acidic_gain_count", "Acidic gained"),
    ReviewAxisMetric("nucleic_acid_facing_proline_glycine_gain_count", "Pro/Gly gained"),
)
NA_FACING_CHEMISTRY_REQUIRED_FIELDS = (
    NA_FACING_CHARGE_FIELD,
    *(metric.field for metric in NA_FACING_CHEMISTRY_METRICS),
)
