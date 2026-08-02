"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/design/planner.py

Deterministic orchestration of the pure TriJunction design stages.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from dnadesign.trijunction.contracts.identity import mapping_sha256
from dnadesign.trijunction.contracts.plan import (
    CheckResult,
    CheckSubject,
    PoolPlan,
    PoolSearchEvidence,
    SelectedJunction,
    TriJunctionPlan,
)
from dnadesign.trijunction.contracts.request import TriJunctionRequest
from dnadesign.trijunction.design.loci import (
    enumerate_loci,
    predict_locus_count,
)
from dnadesign.trijunction.design.matching import JunctionAssignment
from dnadesign.trijunction.design.randomness import derive_seed
from dnadesign.trijunction.design.recovery import merge_universal_recovery_orders, validate_recovery_set
from dnadesign.trijunction.design.resources import (
    estimate_request_workload,
    guard_request_workload,
    guard_uniform_toehold_search,
)
from dnadesign.trijunction.design.search_pipeline import _STRING_SEARCH_V1, _SearchPipeline
from dnadesign.trijunction.design.strands import compose_target
from dnadesign.trijunction.errors import TriJunctionDesignError

_PLAN_SCHEMA = "dnadesign.trijunction.plan.v1"


def design_trijunction(request: TriJunctionRequest) -> TriJunctionPlan:
    """Compile one trusted request into an immutable in-memory plan."""

    return _compile_trijunction(request, pipeline=_STRING_SEARCH_V1)


def _compile_trijunction(
    request: TriJunctionRequest,
    *,
    pipeline: _SearchPipeline,
) -> TriJunctionPlan:
    """Compile with one explicit internal search composition."""

    profile = request.planning
    predicted_loci_by_target: dict[str, int] = {}
    pool_locus_counts_by_id: dict[str, int] = defaultdict(int)
    input_bases = 0
    for target in request.targets:
        locus_count = predict_locus_count(len(target.sequence), profile)
        predicted_loci_by_target[target.id] = locus_count
        pool_locus_counts_by_id[target.pool_id] += locus_count
        input_bases += len(target.sequence)
    missing_loci = sorted(target_id for target_id, locus_count in predicted_loci_by_target.items() if locus_count == 0)
    if missing_loci:
        joined = ", ".join(missing_loci)
        raise TriJunctionDesignError(
            f"Three-way-junction target(s) have no complete toehold locus: {joined}. "
            "Adjust the declared geometry or use a direct-synthesis workflow outside TriJunction."
        )
    request_workload = estimate_request_workload(
        input_bases=input_bases,
        target_count=len(request.targets),
        pool_locus_counts=tuple(pool_locus_counts_by_id[pool_id] for pool_id in sorted(pool_locus_counts_by_id)),
        profile=profile,
    )
    guard_request_workload(request_workload)

    validate_recovery_set(request)
    request_mapping = request.to_mapping()
    request_sha256 = mapping_sha256(request_mapping)
    targets_by_pool: dict[str, list] = defaultdict(list)
    for target in request.targets:
        targets_by_pool[target.pool_id].append(target)

    pool_plans: list[PoolPlan] = []
    assignments_by_pool: dict[str, tuple[JunctionAssignment, ...]] = {}
    checks: list[CheckResult] = []
    for pool_id in sorted(targets_by_pool):
        targets = tuple(sorted(targets_by_pool[pool_id], key=lambda target: target.id))
        guard_uniform_toehold_search(
            locus_count=sum(predicted_loci_by_target[target.id] for target in targets),
            candidates_per_locus=profile.search_range,
            sequence_length=profile.toehold_length,
            iterations=profile.toehold_search_iterations,
        )
        loci_by_target = {target.id: enumerate_loci(target, profile) for target in targets}
        loci = tuple(locus for target_loci in loci_by_target.values() for locus in target_loci)
        toehold_seed = derive_seed(request.seed, pool_id=pool_id, stage="toeholds")
        barcode_generation_seed = derive_seed(request.seed, pool_id=pool_id, stage="barcode-generation")
        barcode_subset_seed = derive_seed(request.seed, pool_id=pool_id, stage="barcode-subsets")
        matching_seed = derive_seed(request.seed, pool_id=pool_id, stage="matching")
        forbidden_toehold_k = profile.barcode_toehold_k
        forbidden_barcode_k = profile.barcode_pair_k
        toehold_selection = pipeline.select_toeholds(
            loci,
            iterations=profile.toehold_search_iterations,
            seed=toehold_seed,
        )
        selected_toeholds = tuple(candidate.sequence for candidate in toehold_selection.candidates)
        candidate_pool = pipeline.generate_barcode_candidates(
            selected_toeholds,
            length=profile.barcode_length,
            count=profile.barcode_pool_factor * len(selected_toeholds),
            forbidden_toehold_k=forbidden_toehold_k,
            forbidden_barcode_k=forbidden_barcode_k,
            gc_min=profile.barcode_gc_min,
            gc_max=profile.barcode_gc_max,
            max_homopolymer=profile.barcode_max_homopolymer,
            max_attempts=profile.barcode_generation_attempts,
            seed=barcode_generation_seed,
        )
        barcode_selection = pipeline.select_barcodes(
            candidate_pool,
            count=len(selected_toeholds),
            iterations=profile.barcode_subset_iterations,
            seed=barcode_subset_seed,
            forbidden_toehold_k=forbidden_toehold_k,
            forbidden_barcode_k=forbidden_barcode_k,
        )
        matching = pipeline.match_barcodes(
            toehold_selection.candidates,
            barcode_selection.barcodes,
            iterations=profile.matching_iterations,
            seed=matching_seed,
        )
        assignments = matching.assignments
        assignments_by_pool[pool_id] = assignments

        per_target_index: dict[str, int] = defaultdict(int)
        selected_junctions: list[SelectedJunction] = []
        for assignment in assignments:
            candidate = assignment.candidate
            per_target_index[candidate.target_id] += 1
            selected_junctions.append(
                SelectedJunction(
                    junction_id=f"{candidate.target_id}:junction-{per_target_index[candidate.target_id]:04d}",
                    target_id=candidate.target_id,
                    pool_id=pool_id,
                    locus_index=candidate.locus_index,
                    candidate_offset=candidate.candidate_offset,
                    start=candidate.start,
                    toehold=candidate.sequence,
                    barcode_id=assignment.barcode_id,
                    barcode=assignment.barcode,
                )
            )
        pool_plans.append(
            PoolPlan(
                pool_id=pool_id,
                junctions=tuple(selected_junctions),
                search=PoolSearchEvidence(
                    pool_id=pool_id,
                    toehold_seed=toehold_seed,
                    barcode_generation_seed=barcode_generation_seed,
                    barcode_subset_seed=barcode_subset_seed,
                    matching_seed=matching_seed,
                    locus_count=len(loci),
                    toehold_paths_evaluated=toehold_selection.paths_evaluated,
                    toehold_min_distance=toehold_selection.minimum_distance,
                    toehold_mean_distance=toehold_selection.mean_distance,
                    toehold_rank_score=toehold_selection.rank_score,
                    barcode_candidates_generated=barcode_selection.candidates_generated,
                    barcode_forbidden_toehold_k=forbidden_toehold_k,
                    barcode_forbidden_barcode_k=forbidden_barcode_k,
                    barcode_subsets_evaluated=barcode_selection.subsets_evaluated,
                    barcode_min_distance=barcode_selection.minimum_distance,
                    barcode_mean_distance=barcode_selection.mean_distance,
                    barcode_rank_score=barcode_selection.rank_score,
                    matchings_evaluated=matching.matchings_evaluated,
                    matching_max_pairwise_lcs=matching.max_pairwise_lcs,
                    thermodynamic_screening="not_run",
                ),
            )
        )
        checks.extend(
            (
                CheckResult(
                    check="physical_pool_one_to_one_matching",
                    status="passed",
                    subject=CheckSubject(kind="pool", id=pool_id),
                    detail=f"{len(assignments)} unique toehold/barcode assignments",
                ),
                CheckResult(
                    check="thermodynamic_screening",
                    status="not_run",
                    subject=CheckSubject(kind="pool", id=pool_id),
                    detail="string checks do not imply thermodynamic orthogonality",
                ),
            )
        )

    target_plans = []
    orders = []
    for target in sorted(request.targets, key=lambda item: (item.pool_id, item.id)):
        composition = compose_target(
            target,
            assignments_by_pool[target.pool_id],
            order_policy=request.order_policy,
        )
        target_plans.append(composition.target)
        orders.extend(composition.orders)
        checks.extend(composition.checks)

    provisional = TriJunctionPlan(
        schema=_PLAN_SCHEMA,
        algorithm=pipeline.algorithm_id,
        request_sha256=request_sha256,
        plan_id="",
        seed=request.seed,
        pools=tuple(pool_plans),
        targets=tuple(target_plans),
        orders=merge_universal_recovery_orders(request, orders),
        checks=tuple(checks),
    )
    plan_id = mapping_sha256(provisional.to_mapping(include_plan_id=False))
    return replace(provisional, plan_id=plan_id)
