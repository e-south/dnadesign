"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_distance_plot_data.py

Distance-matrix data tests for Eco1 RT selection-readiness visuals.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness import (
    mutation_distance_plot,
    sequence_distance_plot,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    SELECTED_PANEL_SIZE,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._plot_rows import (
    candidate_sequence_rows,
    selected_panel_rows,
)


def test_selected_hypothesis_panel_distance_matrix_is_symmetric_with_zero_diagonal() -> None:
    labels, matrix = sequence_distance_plot.build_selected_sequence_distance_matrix(
        panel_rows=selected_panel_rows(),
        candidate_rows=candidate_sequence_rows(),
    )

    assert labels == [f"candidate_{index}" for index in range(1, SELECTED_PANEL_SIZE + 1)]
    assert len(matrix) == SELECTED_PANEL_SIZE
    assert all(len(row) == SELECTED_PANEL_SIZE for row in matrix)
    assert all(matrix[index][index] == 0 for index in range(len(matrix)))
    assert matrix == [list(row) for row in zip(*matrix, strict=True)]
    assert matrix[0][1] == 1
    assert matrix[0][-1] == 7


def test_selected_sequence_distance_matrix_rejects_incomparable_sequences() -> None:
    panel_rows = selected_panel_rows()[:2]
    candidate_rows = candidate_sequence_rows()[:2]
    candidate_rows[1]["sequence"] = ""

    with pytest.raises(ValueError, match="non-empty protein sequence"):
        sequence_distance_plot.build_selected_sequence_distance_matrix(
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
        )

    candidate_rows[1]["sequence"] = "AAA"
    with pytest.raises(ValueError, match="equal-length protein sequences"):
        sequence_distance_plot.build_selected_sequence_distance_matrix(
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
        )


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
