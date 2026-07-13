"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_region_msa_support_plot.py

Region-MSA support plot tests for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness import (
    region_msa_support_plot,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    SELECTED_PANEL_SIZE,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._plot_rows import (
    selected_panel_rows,
)


def test_regionwise_msa_support_matrix_keeps_thumb_track_separate_when_zero() -> None:
    support_rows = []
    for panel_row in selected_panel_rows():
        for region_id, region_label in (
            ("catalytic_or_direct_contact", "Catalytic or direct contact"),
            ("near_retained_dna_rna_region", "Near retained DNA/RNA region"),
            ("thumb_contact_track", "Thumb-contact track"),
            ("c_terminal_primer_rna_recognition_region", "C-terminal primer-RNA recognition region"),
            ("distal_scaffold", "Distal scaffold"),
        ):
            support_rows.append(
                {
                    "candidate_id": panel_row["candidate_id"],
                    "region_id": region_id,
                    "region_label": region_label,
                    "alt_observed_fraction": 0.5 if region_id != "thumb_contact_track" else None,
                    "unobserved_mutation_count": 0,
                    "mutation_count": 1 if region_id != "thumb_contact_track" else 0,
                }
            )

    region_labels, row_labels, matrix, unobserved = region_msa_support_plot.build_selected_region_msa_support_matrix(
        panel_rows=selected_panel_rows(),
        region_msa_support_rows=support_rows,
    )

    assert region_labels == [
        "Catalytic or direct contact",
        "Near retained DNA/RNA region",
        "Thumb-contact track",
        "C-terminal primer-RNA recognition region",
        "Distal scaffold",
    ]
    assert len(row_labels) == SELECTED_PANEL_SIZE
    assert all(len(row) == 5 for row in matrix)
    assert all(row[2] is None for row in matrix)
    assert all(row[2] == 0 for row in unobserved)
