"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_panel_selection_order.py

Panel selection-order tests for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.mutation_distance import (
    nearest_jaccard_distance,
    nearest_shared_count,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_ranking import (
    choose_farthest_candidate as _choose_farthest_candidate,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._panel_contract_fixtures import (
    candidate_row,
)


def test_primary_choice_prefers_lower_basic_loss_before_msa() -> None:
    rows = [
        candidate_row("high_msa_basic_loss", na_facing_mutation_count=1, basic_loss_count=1, msa_fraction=1.0),
        candidate_row("lower_msa_no_basic_loss", na_facing_mutation_count=1, msa_fraction=0.5),
    ]

    chosen, _nearest_distance = _choose_farthest_candidate(
        candidate_rows=rows,
        selected_rows=[],
        sequence_by_id={row["candidate_id"]: "A" * 12 for row in rows},
    )

    assert chosen["candidate_id"] == "lower_msa_no_basic_loss"


def test_first_primary_choice_does_not_reward_mutation_count_before_chemistry() -> None:
    rows = [
        candidate_row(
            "higher_count_with_basic_loss",
            na_facing_mutation_count=40,
            basic_loss_count=1,
            mutation_count_total=60,
        ),
        candidate_row(
            "lower_count_without_basic_loss",
            na_facing_mutation_count=31,
            mutation_count_total=31,
        ),
    ]

    chosen, _nearest_distance = _choose_farthest_candidate(
        candidate_rows=rows,
        selected_rows=[],
        sequence_by_id={row["candidate_id"]: "A" * 12 for row in rows},
    )

    assert chosen["candidate_id"] == "lower_count_without_basic_loss"


def test_first_primary_choice_does_not_reward_higher_total_mutation_count() -> None:
    rows = [
        candidate_row("a_lower_count", na_facing_mutation_count=1, mutation_count_total=31),
        candidate_row("z_higher_count", na_facing_mutation_count=1, mutation_count_total=60),
    ]

    chosen, _nearest_distance = _choose_farthest_candidate(
        candidate_rows=rows,
        selected_rows=[],
        sequence_by_id={row["candidate_id"]: "A" * 12 for row in rows},
    )

    assert chosen["candidate_id"] == "a_lower_count"


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

    chosen, _nearest_distance = _choose_farthest_candidate(
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

    chosen, _nearest_distance = _choose_farthest_candidate(
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

    chosen, _nearest_distance = _choose_farthest_candidate(
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


def test_primary_choice_maximizes_minimum_position_set_distance_before_absolute_coverage() -> None:
    selected = [candidate_row("already_selected", na_facing_mutation_count=0)]
    rows = [
        candidate_row("more_new_positions_but_overlapping", na_facing_mutation_count=5, mutation_count_total=5),
        candidate_row("fewer_new_positions_but_disjoint", na_facing_mutation_count=2, mutation_count_total=2),
    ]

    chosen, _nearest_distance = _choose_farthest_candidate(
        candidate_rows=rows,
        selected_rows=selected,
        sequence_by_id={
            "already_selected": "A" * 12,
            "more_new_positions_but_overlapping": "A" * 11 + "C",
            "fewer_new_positions_but_disjoint": "A" * 10 + "CC",
        },
        mutation_tokens_by_id={
            "already_selected": frozenset({"A10G", "L20V"}),
            "more_new_positions_but_overlapping": frozenset({"A10K", "L20I", "S30T", "S40A", "S50V"}),
            "fewer_new_positions_but_disjoint": frozenset({"A60G", "L70V"}),
        },
        mutation_positions_by_id={
            "already_selected": frozenset({10, 20}),
            "more_new_positions_but_overlapping": frozenset({10, 20, 30, 40, 50}),
            "fewer_new_positions_but_disjoint": frozenset({60, 70}),
        },
    )

    assert chosen["candidate_id"] == "fewer_new_positions_but_disjoint"
    assert chosen["nearest_selected_mutation_position_jaccard_distance"] == 1.0
    assert chosen["new_mutated_position_count_vs_panel"] == 2
    assert chosen["new_exact_substitution_count_vs_panel"] == 2


def test_primary_choice_does_not_mutate_input_triage_rows() -> None:
    selected = [candidate_row("already_selected", na_facing_mutation_count=1)]
    rows = [candidate_row("candidate_a", na_facing_mutation_count=2)]
    original_rows = copy.deepcopy(rows)

    _choose_farthest_candidate(
        candidate_rows=rows,
        selected_rows=selected,
        sequence_by_id={"already_selected": "A" * 12, "candidate_a": "A" * 11 + "C"},
        mutation_tokens_by_id={"already_selected": frozenset({"A10G"}), "candidate_a": frozenset({"A20G"})},
        mutation_positions_by_id={"already_selected": frozenset({10}), "candidate_a": frozenset({20})},
    )

    assert rows == original_rows


def test_nearest_shared_count_uses_the_same_peer_as_nearest_jaccard_distance() -> None:
    candidate = frozenset({1, 2, 3, 4, 5})
    nearest_peer = frozenset({1, 2, 3})
    larger_but_more_distant_overlap = frozenset({1, 2, 3, 4, 6, 7, 8, 9, 10})
    peers = [nearest_peer, larger_but_more_distant_overlap]

    assert nearest_jaccard_distance(candidate, peers) == 0.4
    assert nearest_shared_count(candidate, peers) == 3
