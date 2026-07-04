"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_population_stratification.py

Population-stratification plot tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness import (
    population_stratification,
)


def test_population_stratification_label_positions_separate_clustered_selected_rows() -> None:
    label_positions = population_stratification.selected_label_positions(
        [
            {
                "candidate_id": "a",
                "selection_support_alt_observed_fraction": 1.0,
                "nucleic_acid_facing_mutation_count": 60,
            },
            {
                "candidate_id": "b",
                "selection_support_alt_observed_fraction": 1.0,
                "nucleic_acid_facing_mutation_count": 60,
            },
            {
                "candidate_id": "c",
                "selection_support_alt_observed_fraction": 0.99,
                "nucleic_acid_facing_mutation_count": 58,
            },
            {
                "candidate_id": "d",
                "selection_support_alt_observed_fraction": 0.86,
                "nucleic_acid_facing_mutation_count": 52,
            },
        ],
        y_max=70,
    )

    y_values = [position[1] for position in label_positions.values()]
    assert len(set(y_values)) == len(y_values)
    for upper, lower in zip(sorted(y_values, reverse=True), sorted(y_values, reverse=True)[1:], strict=False):
        assert upper - lower >= 3.8
    assert all(position[0] >= 1.07 for position in label_positions.values())
