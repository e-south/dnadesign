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

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness import (
    NA_FACING_CHEMISTRY_METRICS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.chemistry_balance import (
    build_na_facing_chemistry_balance_matrix,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plots import (
    _class_percentile,
    build_selected_sequence_distance_matrix,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.regional_plots import (
    build_regional_mutation_burden_matrix,
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


def test_class_local_moderate_percentile_prefers_class_median() -> None:
    assert _class_percentile(selected_value=40.0, class_values=[0.0, 40.0, 90.0], direction="moderate") == 100.0
    assert _class_percentile(selected_value=90.0, class_values=[0.0, 40.0, 90.0], direction="moderate") == 0.0


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


def test_regional_mutation_burden_fallback_does_not_treat_conservation_as_contact() -> None:
    _region_labels, _row_labels, matrix = build_regional_mutation_burden_matrix(
        panel_rows=[
            {
                "candidate_id": "candidate_1",
                "design_class_id": ALL_SPECS[0].design_class_id,
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

    assert matrix == [[0, 1, 0, 1]]


def test_na_facing_chemistry_balance_matrix_uses_selected_triage_rows() -> None:
    row_labels, charge_delta, metric_labels, matrix = build_na_facing_chemistry_balance_matrix(
        panel_rows=_panel_rows(),
        triage_rows=[
            {
                "candidate_id": f"candidate_{index}",
                "design_class_id": spec.design_class_id,
                "nucleic_acid_facing_charge_delta": index - 3,
                "nucleic_acid_facing_basic_gain_count": index,
                "nucleic_acid_facing_basic_loss_count": 6 - index,
                "nucleic_acid_facing_acidic_gain_count": index % 2,
                "nucleic_acid_facing_proline_glycine_gain_count": index % 3,
            }
            for index, spec in enumerate(ALL_SPECS, start=1)
        ],
    )

    assert len(row_labels) == len(ALL_SPECS)
    assert charge_delta == [-2, -1, 0, 1, 2, 3]
    assert metric_labels == [metric.label for metric in NA_FACING_CHEMISTRY_METRICS]
    assert len(matrix) == len(ALL_SPECS)
    assert matrix[0] == [1, 5, 1, 1]


def test_na_facing_chemistry_balance_matrix_fails_on_missing_selected_field() -> None:
    triage_rows = [
        {
            "candidate_id": f"candidate_{index}",
            "design_class_id": spec.design_class_id,
            "nucleic_acid_facing_charge_delta": index - 3,
            "nucleic_acid_facing_basic_gain_count": index,
            "nucleic_acid_facing_basic_loss_count": 6 - index,
            "nucleic_acid_facing_acidic_gain_count": index % 2,
            "nucleic_acid_facing_proline_glycine_gain_count": index % 3,
        }
        for index, spec in enumerate(ALL_SPECS, start=1)
    ]
    triage_rows[0].pop("nucleic_acid_facing_basic_loss_count")
    triage_rows[1]["nucleic_acid_facing_charge_delta"] = None

    with pytest.raises(ValueError) as error:
        build_na_facing_chemistry_balance_matrix(panel_rows=_panel_rows(), triage_rows=triage_rows)

    message = str(error.value)
    assert "candidate_1" in message
    assert "nucleic_acid_facing_basic_loss_count" in message
