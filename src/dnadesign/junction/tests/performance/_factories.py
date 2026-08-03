"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/performance/_factories.py

Shared test factories for junction performance contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.junction.contracts import PlanningProfile
from dnadesign.junction.design.loci import ToeholdCandidate


def candidate(index: int, sequence: str) -> ToeholdCandidate:
    return ToeholdCandidate(
        target_id="target",
        assembly_group_id="assembly",
        locus_index=index,
        candidate_offset=0,
        start=index,
        sequence=sequence,
    )


def planning_profile() -> PlanningProfile:
    return PlanningProfile(
        nominal_fragment_oligo_length=46,
        barcode_length=16,
        toehold_length=8,
        search_range=2,
        toehold_search_iterations=40,
        barcode_pool_factor=5,
        barcode_generation_attempts=100_000,
        barcode_toehold_k=4,
        barcode_pair_k=5,
        barcode_subset_iterations=40,
        matching_iterations=100,
        barcode_gc_min=0.25,
        barcode_gc_max=0.75,
        barcode_max_homopolymer=3,
    )
