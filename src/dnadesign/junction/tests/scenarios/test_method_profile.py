"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/scenarios/test_method_profile.py

Bounded evidence for literature-starting parameters and v1 deviations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.junction import parse_request, plan
from dnadesign.junction.design.loci import enumerate_loci, predict_locus_count
from dnadesign.junction.tests.scenarios.factories import scale_request_mapping


def _paper_starting_request(*, target_length: int) -> dict[str, object]:
    mapping = scale_request_mapping(
        target_count=1,
        target_length=target_length,
        topology="shared",
        oligo_length=96,
        search_range=15,
        barcode_generation_attempts=500_000,
    )
    planning = mapping["planning"]
    assert isinstance(planning, dict)
    planning.update(
        toehold_search_iterations=1_000,
        barcode_toehold_k=5,
        barcode_pair_k=6,
        barcode_subset_iterations=100,
        matching_iterations=100,
    )
    return mapping


def test_v1_terminal_locus_policy_keeps_orders_inside_the_declared_ceiling() -> None:
    request = parse_request(_paper_starting_request(target_length=180))

    loci = enumerate_loci(request.targets[0], request.planning)
    result = plan(request)

    # The pooled paper's literal next-locus break would stop after start 64.
    # V1 retains start 116 so the terminal barcode-bearing order stays inside
    # the explicit L + R - 1 ceiling instead of growing to 128 nt.
    assert [locus.candidates[0].start for locus in loci] == [64, 116]
    assembly_orders = [order for order in result.orders if order.fragment_id is not None]
    assert max(order.length for order in assembly_orders) <= request.order_policy.max_oligo_length


@pytest.mark.parametrize(
    ("target_length", "expected_loci"),
    [(87, 0), (88, 1), (148, 1), (149, 2), (190, 2), (200, 2), (201, 3)],
)
def test_v1_terminal_locus_transition_boundaries(target_length: int, expected_loci: int) -> None:
    request = parse_request(_paper_starting_request(target_length=max(target_length, 88)))

    assert predict_locus_count(target_length, request.planning) == expected_loci


@pytest.mark.slow
def test_paper_starting_profile_runs_a_meaningful_bounded_search() -> None:
    request = parse_request(_paper_starting_request(target_length=500))

    result = plan(request)
    search = result.assembly_groups[0].search

    assert len(result.assembly_groups[0].junctions) == 8
    assert search.toehold_paths_evaluated > 100
    assert search.barcode_candidates_generated == 40
    assert search.barcode_forbidden_toehold_k == 5
    assert search.barcode_forbidden_barcode_k == 6
    assert search.barcode_subsets_evaluated > 100
    assert search.matchings_evaluated > 100
