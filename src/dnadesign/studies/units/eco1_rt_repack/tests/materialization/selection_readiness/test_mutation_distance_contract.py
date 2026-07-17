"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_mutation_distance_contract.py

Mutation-distance contract tests for Eco1 RT panel selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.mutation_distance import (
    nearest_jaccard_distance,
    nearest_shared_count,
)


def test_nearest_shared_count_uses_the_same_peer_as_nearest_jaccard_distance() -> None:
    candidate = frozenset({1, 2, 3, 4, 5})
    nearest_peer = frozenset({1, 2, 3})
    larger_but_more_distant_overlap = frozenset({1, 2, 3, 4, 6, 7, 8, 9, 10})
    peers = [nearest_peer, larger_but_more_distant_overlap]

    assert nearest_jaccard_distance(candidate, peers) == 0.4
    assert nearest_shared_count(candidate, peers) == 3
