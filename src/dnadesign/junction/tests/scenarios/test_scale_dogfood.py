"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/scenarios/test_scale_dogfood.py

End-to-end scale evidence for supported and refused request topologies.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnadesign.junction import build, parse_request, plan, verify
from dnadesign.junction.errors import JunctionDesignError
from dnadesign.junction.tests.scenarios.factories import scale_request_mapping


def _bundle_bytes(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.mark.parametrize("target_length", [1_000, 10_000])
def test_single_target_bounded_search_bundle_round_trip_at_declared_lengths(
    tmp_path: Path,
    target_length: int,
) -> None:
    request = parse_request(
        scale_request_mapping(
            target_count=1,
            target_length=target_length,
            topology="shared",
            nominal_fragment_oligo_length=96,
            search_range=2,
            barcode_generation_attempts=250_000,
        )
    )
    destination = tmp_path / f"target-{target_length}"

    published = build(request, destination=destination)
    verified = verify(destination)
    result = plan(request)

    assert verified.plan_id == published.plan_id == result.plan_id
    assert result.targets[0].reconstructed_target == request.targets[0].sequence
    assert max(order.length for order in result.orders) <= request.order_policy.max_oligo_length
    assert result.targets[0].recovery.forward_five_prime_extension == "GGTCTCA"
    assert result.targets[0].recovery.reverse_five_prime_extension == "CGTCTCA"


def test_bundle_bytes_do_not_depend_on_destination(tmp_path: Path) -> None:
    request = parse_request(
        scale_request_mapping(
            target_count=1,
            target_length=1_000,
            topology="shared",
            nominal_fragment_oligo_length=96,
            search_range=2,
            barcode_generation_attempts=250_000,
        )
    )
    first = tmp_path / "first" / "bundle"
    second = tmp_path / "second" / "bundle"

    first_publication = build(request, destination=first)
    second_publication = build(request, destination=second)

    assert first_publication.plan_id == second_publication.plan_id
    assert _bundle_bytes(first) == _bundle_bytes(second)
    assert verify(first).plan_id == verify(second).plan_id


@pytest.mark.slow
def test_shared_100_target_bounded_search_load_shape_is_jointly_planned() -> None:
    request = parse_request(
        scale_request_mapping(
            target_count=100,
            target_length=1_000,
            topology="shared",
            nominal_fragment_oligo_length=200,
            search_range=2,
            barcode_generation_attempts=500_000,
        )
    )

    result = plan(request)

    assert len(result.targets) == 100
    assert len(result.assembly_groups) == 1
    assert len(result.assembly_groups[0].junctions) == 600
    assert result.assembly_groups[0].search.toehold_paths_evaluated > 1
    assert result.assembly_groups[0].search.barcode_subsets_evaluated > 1
    assert result.assembly_groups[0].search.matchings_evaluated > 1
    assert all(
        target.reconstructed_target == source.sequence for target, source in zip(result.targets, request.targets)
    )


@pytest.mark.slow
def test_independent_1000_target_bundle_round_trip_stays_within_bounded_load_shape(tmp_path: Path) -> None:
    request = parse_request(
        scale_request_mapping(
            target_count=1_000,
            target_length=1_000,
            topology="independent",
            nominal_fragment_oligo_length=200,
            search_range=2,
            barcode_generation_attempts=500,
        )
    )
    destination = tmp_path / "independent-1000"

    published = build(request, destination=destination)
    verified = verify(destination)
    plan_payload = json.loads((destination / "plan.json").read_bytes())

    assert published.plan_id == verified.plan_id == plan_payload["plan_id"]
    assert len(plan_payload["targets"]) == 1_000
    assert len(plan_payload["assembly_groups"]) == 1_000
    assert sum(len(group["junctions"]) for group in plan_payload["assembly_groups"]) == 6_000
    assert all(group["search"]["toehold_paths_evaluated"] > 1 for group in plan_payload["assembly_groups"])
    assert all(group["search"]["barcode_subsets_evaluated"] > 1 for group in plan_payload["assembly_groups"])
    assert all(group["search"]["matchings_evaluated"] >= 1 for group in plan_payload["assembly_groups"])


def test_1000_target_shared_assembly_group_fails_closed_without_automatic_chunking() -> None:
    request = parse_request(
        scale_request_mapping(
            target_count=1_000,
            target_length=1_000,
            topology="shared",
            nominal_fragment_oligo_length=200,
            search_range=1,
            barcode_generation_attempts=500_000,
        )
    )

    with pytest.raises(JunctionDesignError, match="Request-wide barcode-subset cache bytes"):
        plan(request)
