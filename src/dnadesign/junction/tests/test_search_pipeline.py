"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/test_search_pipeline.py

Internal search-composition seam and public-lifecycle boundaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

import dnadesign.junction as junction
from dnadesign.junction import publication as publication_module
from dnadesign.junction.contracts import parse_request
from dnadesign.junction.design import search_pipeline as search_pipeline_module
from dnadesign.junction.design.barcodes import BarcodeSelection
from dnadesign.junction.design.loci import ToeholdCandidate, ToeholdLocus
from dnadesign.junction.design.matching import MatchingSelection
from dnadesign.junction.design.planner import _compile_junction, design_junction
from dnadesign.junction.design.randomness import derive_seed
from dnadesign.junction.design.search_pipeline import _STRING_SEARCH_V1, _SearchPipeline
from dnadesign.junction.design.toeholds import ToeholdSelection
from dnadesign.junction.tests.scenarios.factories import scale_request_mapping
from dnadesign.junction.tests.test_planner import _request_mapping


def test_internal_compile_uses_each_search_stage_and_pipeline_identity() -> None:
    calls: list[tuple[str, int]] = []

    def select_toeholds(
        loci: tuple[ToeholdLocus, ...],
        *,
        iterations: int,
        seed: int,
    ) -> ToeholdSelection:
        calls.append(("select_toeholds", seed))
        return _STRING_SEARCH_V1.select_toeholds(loci, iterations=iterations, seed=seed)

    def generate_barcode_candidates(
        toeholds: tuple[str, ...],
        *,
        length: int,
        count: int,
        forbidden_toehold_k: int,
        forbidden_barcode_k: int,
        gc_min: float,
        gc_max: float,
        max_homopolymer: int,
        max_attempts: int,
        seed: int,
    ) -> tuple[str, ...]:
        calls.append(("generate_barcode_candidates", seed))
        return _STRING_SEARCH_V1.generate_barcode_candidates(
            toeholds,
            length=length,
            count=count,
            forbidden_toehold_k=forbidden_toehold_k,
            forbidden_barcode_k=forbidden_barcode_k,
            gc_min=gc_min,
            gc_max=gc_max,
            max_homopolymer=max_homopolymer,
            max_attempts=max_attempts,
            seed=seed,
        )

    def select_barcodes(
        candidates: tuple[str, ...],
        *,
        count: int,
        iterations: int,
        seed: int,
        forbidden_toehold_k: int,
        forbidden_barcode_k: int,
    ) -> BarcodeSelection:
        calls.append(("select_barcodes", seed))
        return _STRING_SEARCH_V1.select_barcodes(
            candidates,
            count=count,
            iterations=iterations,
            seed=seed,
            forbidden_toehold_k=forbidden_toehold_k,
            forbidden_barcode_k=forbidden_barcode_k,
        )

    def match_barcodes(
        candidates: tuple[ToeholdCandidate, ...],
        barcodes: tuple[str, ...],
        *,
        iterations: int,
        seed: int,
    ) -> MatchingSelection:
        calls.append(("match_barcodes", seed))
        return _STRING_SEARCH_V1.match_barcodes(
            candidates,
            barcodes,
            iterations=iterations,
            seed=seed,
        )

    pipeline = replace(
        _STRING_SEARCH_V1,
        algorithm_id="dnadesign.junction.test-spy.v1",
        select_toeholds=select_toeholds,
        generate_barcode_candidates=generate_barcode_candidates,
        select_barcodes=select_barcodes,
        match_barcodes=match_barcodes,
    )

    request = parse_request(_request_mapping())
    plan = _compile_junction(request, pipeline=pipeline)
    canonical = design_junction(request)

    assert calls == [
        ("select_toeholds", derive_seed(request.seed, assembly_group_id="assembly-a", stage="toeholds")),
        (
            "generate_barcode_candidates",
            derive_seed(request.seed, assembly_group_id="assembly-a", stage="barcode-generation"),
        ),
        ("select_barcodes", derive_seed(request.seed, assembly_group_id="assembly-a", stage="barcode-subsets")),
        ("match_barcodes", derive_seed(request.seed, assembly_group_id="assembly-a", stage="matching")),
    ]
    assert plan.algorithm == pipeline.algorithm_id
    assert canonical.algorithm == _STRING_SEARCH_V1.algorithm_id
    assert replace(plan, algorithm=canonical.algorithm, plan_id=canonical.plan_id) == canonical


def test_public_planner_is_hermetic_and_does_not_export_search_composition() -> None:
    assert tuple(inspect.signature(design_junction).parameters) == ("request",)
    assert "_SearchPipeline" not in junction.__all__
    assert "_STRING_SEARCH_V1" not in junction.__all__
    assert search_pipeline_module.__all__ == ()
    assert [name for name, value in vars(search_pipeline_module).items() if isinstance(value, _SearchPipeline)] == [
        "_STRING_SEARCH_V1"
    ]


def test_string_v1_has_a_fixed_conformance_vector() -> None:
    result = design_junction(parse_request(_request_mapping()))

    assert result.algorithm == "dnadesign.junction.string.v1"
    assert result.request_sha256 == "sha256:3bef4866d8088dd2b546503757b1cfca1ccd6c1d0621c9328c1d30a44c28b724"
    assert result.plan_id == "sha256:dba5411822eec8559a9be5c85e67554253b3b9da3065336e4e2e5331151be4a0"


def test_publication_package_does_not_export_an_alternate_lifecycle() -> None:
    assert publication_module.__all__ == ["BundleVerification", "PublishedJunctionBundle"]
    assert not hasattr(publication_module, "preflight_bundle_destination")
    assert not hasattr(publication_module, "publish_bundle")
    assert not hasattr(publication_module, "verify_bundle")


def test_v1_toehold_selection_jointly_receives_every_target_in_an_assembly_group() -> None:
    observed_target_sets: list[set[str]] = []

    def select_toeholds(
        loci: tuple[ToeholdLocus, ...],
        *,
        iterations: int,
        seed: int,
    ) -> ToeholdSelection:
        observed_target_sets.append({locus.target_id for locus in loci})
        return _STRING_SEARCH_V1.select_toeholds(loci, iterations=iterations, seed=seed)

    pipeline = replace(_STRING_SEARCH_V1, select_toeholds=select_toeholds)
    request = parse_request(
        scale_request_mapping(
            target_count=2,
            target_length=1_000,
            topology="shared",
            oligo_length=96,
            search_range=2,
            barcode_generation_attempts=250_000,
        )
    )

    _compile_junction(request, pipeline=pipeline)

    assert observed_target_sets == [{"target-0000", "target-0001"}]


def test_search_pipeline_contract_fails_fast_for_invalid_composition() -> None:
    with pytest.raises(ValueError, match="algorithm_id"):
        replace(_STRING_SEARCH_V1, algorithm_id=" invalid id ")
    with pytest.raises(TypeError, match="match_barcodes"):
        replace(_STRING_SEARCH_V1, match_barcodes=None)  # type: ignore[arg-type]
