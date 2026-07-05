"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_plot_data.py

Plot-data tests for Eco1 RT selection-readiness visuals.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plots import (
    build_regional_mutation_burden_matrix,
    build_selected_sequence_distance_matrix,
)


def _panel_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"candidate_{index}",
            "design_class_id": spec.design_class_id,
        }
        for index, spec in enumerate(ALL_SPECS, start=1)
    ]


def _candidate_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"candidate_{index}",
            "sequence": "A" * index + "C" * (8 - index),
            "canonical_mutations": [f"A{index}G", f"L{index + 20}V"],
        }
        for index in range(1, len(ALL_SPECS) + 1)
    ]


def test_selected_six_distance_matrix_is_symmetric_with_zero_diagonal() -> None:
    labels, matrix = build_selected_sequence_distance_matrix(
        panel_rows=_panel_rows(),
        candidate_rows=_candidate_rows(),
    )

    assert labels == [f"candidate_{index}" for index in range(1, len(ALL_SPECS) + 1)]
    assert len(matrix) == len(ALL_SPECS)
    assert all(len(row) == len(ALL_SPECS) for row in matrix)
    assert all(matrix[index][index] == 0 for index in range(len(matrix)))
    assert matrix == [list(row) for row in zip(*matrix, strict=True)]


def test_regional_mutation_burden_matrix_handles_six_selected_rows() -> None:
    region_labels, row_labels, matrix = build_regional_mutation_burden_matrix(
        panel_rows=_panel_rows(),
        candidate_rows=_candidate_rows(),
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
        "Near retained DNA/RNA annulus",
        "Thumb-contact track",
        "Distal scaffold",
    ]
    assert len(row_labels) == len(ALL_SPECS)
    assert len(matrix) == len(ALL_SPECS)
    assert all(len(row) == len(region_labels) for row in matrix)
