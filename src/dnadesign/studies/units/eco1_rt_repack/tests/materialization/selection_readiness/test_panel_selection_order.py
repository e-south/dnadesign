"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_panel_selection_order.py

Panel selection-order tests for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel import (
    _choose_primary_candidate,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._panel_contract_fixtures import (
    candidate_row,
)


def test_primary_choice_prefers_lower_basic_loss_before_msa() -> None:
    rows = [
        candidate_row("high_msa_basic_loss", na_facing_mutation_count=1, basic_loss_count=1, msa_fraction=1.0),
        candidate_row("lower_msa_no_basic_loss", na_facing_mutation_count=1, msa_fraction=0.5),
    ]

    chosen, _nearest_distance = _choose_primary_candidate(
        candidate_rows=rows,
        selected_rows=[],
        sequence_by_id={row["candidate_id"]: "A" * 12 for row in rows},
    )

    assert chosen["candidate_id"] == "lower_msa_no_basic_loss"


def test_primary_choice_prefers_lower_proline_glycine_gain_before_msa() -> None:
    rows = [
        candidate_row(
            "high_msa_with_proline_gain",
            na_facing_mutation_count=2,
            proline_glycine_gain_count=1,
            msa_fraction=1.0,
        ),
        candidate_row("lower_msa_no_proline_gain", na_facing_mutation_count=2, msa_fraction=0.5),
    ]

    chosen, _nearest_distance = _choose_primary_candidate(
        candidate_rows=rows,
        selected_rows=[],
        sequence_by_id={row["candidate_id"]: "A" * 12 for row in rows},
    )

    assert chosen["candidate_id"] == "lower_msa_no_proline_gain"


def test_primary_choice_prefers_lower_c_terminal_rmsd_after_chemistry_ties() -> None:
    rows = [
        candidate_row("higher_c_terminal_rmsd", na_facing_mutation_count=40, c_terminal_rmsd=2.0),
        candidate_row("lower_c_terminal_rmsd", na_facing_mutation_count=40, c_terminal_rmsd=1.2),
    ]

    chosen, _nearest_distance = _choose_primary_candidate(
        candidate_rows=rows,
        selected_rows=[],
        sequence_by_id={row["candidate_id"]: "A" * 12 for row in rows},
    )

    assert chosen["candidate_id"] == "lower_c_terminal_rmsd"


def test_primary_choice_prefers_mutation_dissimilarity_before_local_rmsd_micro_differences() -> None:
    selected = [candidate_row("already_selected", na_facing_mutation_count=0)]
    rows = [
        candidate_row("lower_rmsd_overlapping_mutations", na_facing_mutation_count=0, c_terminal_rmsd=1.0),
        candidate_row("higher_rmsd_distinct_mutations", na_facing_mutation_count=0, c_terminal_rmsd=1.6),
    ]

    chosen, _nearest_distance = _choose_primary_candidate(
        candidate_rows=rows,
        selected_rows=selected,
        sequence_by_id={
            "already_selected": "A" * 12,
            "lower_rmsd_overlapping_mutations": "A" * 11 + "C",
            "higher_rmsd_distinct_mutations": "A" * 10 + "CC",
        },
        mutation_tokens_by_id={
            "already_selected": frozenset({"A10G", "L20V"}),
            "lower_rmsd_overlapping_mutations": frozenset({"A10G", "L20V"}),
            "higher_rmsd_distinct_mutations": frozenset({"A30G", "L40V"}),
        },
        mutation_positions_by_id={
            "already_selected": frozenset({10, 20}),
            "lower_rmsd_overlapping_mutations": frozenset({10, 20}),
            "higher_rmsd_distinct_mutations": frozenset({30, 40}),
        },
    )

    assert chosen["candidate_id"] == "higher_rmsd_distinct_mutations"


def test_primary_choice_allows_distinct_rows_before_soft_chemistry_penalties() -> None:
    selected = [candidate_row("already_selected", na_facing_mutation_count=0)]
    rows = [
        candidate_row("overlapping_no_soft_warning", na_facing_mutation_count=0),
        candidate_row(
            "distinct_with_soft_warning",
            na_facing_mutation_count=3,
            basic_loss_count=1,
            proline_glycine_gain_count=1,
        ),
    ]

    chosen, _nearest_distance = _choose_primary_candidate(
        candidate_rows=rows,
        selected_rows=selected,
        sequence_by_id={
            "already_selected": "A" * 12,
            "overlapping_no_soft_warning": "A" * 11 + "C",
            "distinct_with_soft_warning": "A" * 10 + "CC",
        },
        mutation_tokens_by_id={
            "already_selected": frozenset({"A10G", "L20V"}),
            "overlapping_no_soft_warning": frozenset({"A10G", "L20V"}),
            "distinct_with_soft_warning": frozenset({"A30K", "L40R"}),
        },
        mutation_positions_by_id={
            "already_selected": frozenset({10, 20}),
            "overlapping_no_soft_warning": frozenset({10, 20}),
            "distinct_with_soft_warning": frozenset({30, 40}),
        },
    )

    assert chosen["candidate_id"] == "distinct_with_soft_warning"
