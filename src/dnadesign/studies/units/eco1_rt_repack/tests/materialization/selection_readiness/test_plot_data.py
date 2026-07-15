"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_plot_data.py

Plot-data tests for Eco1 RT selection-readiness visuals.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness import (
    NA_FACING_CHEMISTRY_METRICS,
    mutation_distance_plot,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.chemistry_balance import (
    build_na_facing_chemistry_balance_matrix,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    SELECTED_PANEL_SIZE,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.regional_plots import (
    build_regional_mutation_burden_matrix,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    CURRENT_SELECTION_PLOTS,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._plot_rows import (
    candidate_sequence_rows,
    selected_panel_rows,
    selected_triage_rows,
)


def test_selected_hypothesis_panel_distance_matrix_is_symmetric_with_zero_diagonal() -> None:
    labels, matrix = mutation_distance_plot.build_selected_sequence_distance_matrix(
        panel_rows=selected_panel_rows(),
        candidate_rows=candidate_sequence_rows(),
    )

    assert labels == [f"candidate_{index}" for index in range(1, SELECTED_PANEL_SIZE + 1)]
    assert len(matrix) == SELECTED_PANEL_SIZE
    assert all(len(row) == SELECTED_PANEL_SIZE for row in matrix)
    assert all(matrix[index][index] == 0 for index in range(len(matrix)))
    assert matrix == [list(row) for row in zip(*matrix, strict=True)]


def test_selected_mutation_dissimilarity_matrices_are_symmetric_with_zero_diagonal() -> None:
    labels, position_matrix, token_matrix = mutation_distance_plot.build_selected_mutation_dissimilarity_matrices(
        panel_rows=selected_panel_rows(),
        candidate_rows=candidate_sequence_rows(),
    )

    assert labels == [f"candidate_{index}" for index in range(1, SELECTED_PANEL_SIZE + 1)]
    assert len(position_matrix) == SELECTED_PANEL_SIZE
    assert len(token_matrix) == SELECTED_PANEL_SIZE
    assert all(position_matrix[index][index] == 0 for index in range(len(position_matrix)))
    assert all(token_matrix[index][index] == 0 for index in range(len(token_matrix)))
    assert position_matrix == [list(row) for row in zip(*position_matrix, strict=True)]
    assert token_matrix == [list(row) for row in zip(*token_matrix, strict=True)]


def test_mutation_distance_context_keeps_generation_policies_separate() -> None:
    candidate_rows = [
        {"candidate_id": "d1", "policy_id": DISTAL_SCAFFOLD_POLICY_ID, "canonical_mutations": ["A10G"]},
        {"candidate_id": "d2", "policy_id": DISTAL_SCAFFOLD_POLICY_ID, "canonical_mutations": ["A20G"]},
        {
            "candidate_id": "p1",
            "policy_id": NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
            "canonical_mutations": ["A30G"],
        },
        {
            "candidate_id": "p2",
            "policy_id": NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
            "canonical_mutations": ["A30G", "A40G"],
        },
    ]
    panel_rows = [
        {"candidate_id": "d1", "policy_id": DISTAL_SCAFFOLD_POLICY_ID, "selection_rank": 1},
        {"candidate_id": "d2", "policy_id": DISTAL_SCAFFOLD_POLICY_ID, "selection_rank": 2},
        {"candidate_id": "p1", "policy_id": NEAR_DNA_RNA_ACID_FREE_POLICY_ID, "selection_rank": 3},
        {"candidate_id": "p2", "policy_id": NEAR_DNA_RNA_ACID_FREE_POLICY_ID, "selection_rank": 4},
    ]
    triage_rows = [{"candidate_id": row["candidate_id"], "selection_contract_pass": True} for row in candidate_rows]

    context = mutation_distance_plot.build_within_policy_position_distance_context(
        panel_rows=panel_rows,
        candidate_rows=candidate_rows,
        triage_rows=triage_rows,
    )

    assert set(context) == {DISTAL_SCAFFOLD_POLICY_ID, NEAR_DNA_RNA_ACID_FREE_POLICY_ID}
    assert context[DISTAL_SCAFFOLD_POLICY_ID]["candidate_pair_distances"] == [1.0]
    assert context[DISTAL_SCAFFOLD_POLICY_ID]["selected_pair_distances"] == [1.0]
    assert context[NEAR_DNA_RNA_ACID_FREE_POLICY_ID]["candidate_pair_distances"] == [0.5]
    assert context[NEAR_DNA_RNA_ACID_FREE_POLICY_ID]["selected_pair_distances"] == [0.5]


def test_regional_mutation_burden_matrix_handles_eight_selected_rows() -> None:
    region_labels, row_labels, matrix = build_regional_mutation_burden_matrix(
        panel_rows=selected_panel_rows(),
        candidate_rows=candidate_sequence_rows(),
        mask_residues=[
            {"canonical_position": 1, "protected": True, "wang_ec86_direct_contact_prior": False},
            {"canonical_position": 2, "protected": False, "wang_ec86_direct_contact_prior": True},
            {"canonical_position": 3, "protected": False, "distance_to_retained_na_angstrom": 8.0},
            {"canonical_position": 4, "protected": False, "distance_to_retained_na_angstrom": 20.0},
            {"canonical_position": 21, "protected": False, "distance_to_retained_na_angstrom": 20.0},
            {"canonical_position": 22, "protected": False, "distance_to_retained_na_angstrom": 20.0},
            {"canonical_position": 23, "protected": False, "distance_to_retained_na_angstrom": 20.0},
            {"canonical_position": 24, "protected": False, "distance_to_retained_na_angstrom": 20.0},
            {"canonical_position": 25, "protected": False, "distance_to_retained_na_angstrom": 20.0},
            {"canonical_position": 26, "protected": False, "distance_to_retained_na_angstrom": 20.0},
        ],
    )

    assert region_labels == [
        "Catalytic or direct contact",
        "Near retained DNA/RNA region",
        "Thumb-contact track",
        "Designable C-terminal boundary 230-254",
        "Fixed C-terminal context 255-311",
        "Distal scaffold",
    ]
    assert len(row_labels) == SELECTED_PANEL_SIZE
    assert len(matrix) == SELECTED_PANEL_SIZE
    assert all(len(row) == len(region_labels) for row in matrix)


def test_regional_mutation_burden_fallback_does_not_treat_conservation_as_contact() -> None:
    _region_labels, _row_labels, matrix = build_regional_mutation_burden_matrix(
        panel_rows=[
            {
                "candidate_id": "candidate_1",
                "policy_id": COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
            }
        ],
        candidate_rows=[
            {
                "candidate_id": "candidate_1",
                "canonical_mutations": ["A10G", "L11V"],
            }
        ],
        mask_residues=[
            {
                "canonical_position": 10,
                "protected": True,
                "motif_protected": False,
                "wang_ec86_direct_contact_prior": False,
                "direct_retained_dna_rna_contact_5a": False,
                "distance_to_retained_na_angstrom": 8.0,
            },
            {
                "canonical_position": 11,
                "protected": True,
                "motif_protected": False,
                "wang_ec86_direct_contact_prior": False,
                "direct_retained_dna_rna_contact_5a": False,
                "distance_to_retained_na_angstrom": 20.0,
            },
        ],
    )

    assert matrix == [[0, 1, 0, 0, 0, 1]]


def test_na_facing_chemistry_balance_matrix_uses_selected_triage_rows() -> None:
    row_labels, charge_delta, metric_labels, matrix = build_na_facing_chemistry_balance_matrix(
        panel_rows=selected_panel_rows(),
        triage_rows=selected_triage_rows(),
    )

    assert len(row_labels) == SELECTED_PANEL_SIZE
    assert charge_delta == [-2, -1, 0, 1, 2, 3, 4, 5]
    assert metric_labels == [metric.label for metric in NA_FACING_CHEMISTRY_METRICS]
    assert len(matrix) == SELECTED_PANEL_SIZE
    assert matrix[0] == [1, 5, 1, 1]


def test_na_facing_chemistry_balance_matrix_fails_on_missing_selected_field() -> None:
    triage_rows = selected_triage_rows()
    triage_rows[0].pop("nucleic_acid_facing_basic_loss_count")
    triage_rows[1]["nucleic_acid_facing_charge_delta"] = None

    with pytest.raises(ValueError) as error:
        build_na_facing_chemistry_balance_matrix(panel_rows=selected_panel_rows(), triage_rows=triage_rows)

    message = str(error.value)
    assert "candidate_1" in message
    assert "nucleic_acid_facing_basic_loss_count" in message


def test_selection_plot_funnel_stage_ids_use_public_trace_ontology() -> None:
    assert {plot.funnel_stage_id for plot in CURRENT_SELECTION_PLOTS} <= {
        "",
        "local_geometry_screen",
        "selected_panel",
    }
