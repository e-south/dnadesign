"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/design/resources/__init__.py

Stable facade for versioned junction workload policy and guards.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .batching import (
    pair_lookup_scratch_bytes,
    pair_selection_fixed_scratch_bytes,
    pair_selection_reduction_chunk_size,
    pair_selection_reduction_scratch_bytes,
    upper_triangle_index_batches,
)
from .estimates import (
    RequestWorkloadEstimate,
    barcode_distance_cache_bytes,
    barcode_generation_state_bytes,
    capped_toehold_path_count,
    estimate_request_workload,
    estimated_matching_substring_character_visits,
    estimated_toehold_distance_lookups,
    estimated_toehold_dp_cells,
    kmer_set_state_bytes,
    sampled_barcode_subset_state_bytes,
    sampled_matching_state_bytes,
    sampled_toehold_search_state_bytes,
    toehold_distance_cache_bytes,
)
from .guards import (
    guard_barcode_generation,
    guard_barcode_subset_search,
    guard_matching_search,
    guard_request_workload,
    guard_uniform_toehold_search,
)
from .limits import (
    MAX_BARCODE_DISTANCE_SEQUENCE_LENGTH,
    MAX_PAIR_DISTANCE_SCRATCH_BYTES,
    MAX_REQUEST_BARCODE_GENERATION_BASE_VISITS,
    MAX_TOEHOLD_CACHE_BYTES,
    REQUEST_WORKLOAD_POLICY,
)

__all__ = [
    "MAX_BARCODE_DISTANCE_SEQUENCE_LENGTH",
    "MAX_PAIR_DISTANCE_SCRATCH_BYTES",
    "MAX_REQUEST_BARCODE_GENERATION_BASE_VISITS",
    "MAX_TOEHOLD_CACHE_BYTES",
    "REQUEST_WORKLOAD_POLICY",
    "RequestWorkloadEstimate",
    "barcode_distance_cache_bytes",
    "barcode_generation_state_bytes",
    "capped_toehold_path_count",
    "estimate_request_workload",
    "estimated_matching_substring_character_visits",
    "estimated_toehold_distance_lookups",
    "estimated_toehold_dp_cells",
    "guard_barcode_generation",
    "guard_barcode_subset_search",
    "guard_matching_search",
    "guard_request_workload",
    "guard_uniform_toehold_search",
    "kmer_set_state_bytes",
    "pair_lookup_scratch_bytes",
    "pair_selection_fixed_scratch_bytes",
    "pair_selection_reduction_chunk_size",
    "pair_selection_reduction_scratch_bytes",
    "sampled_barcode_subset_state_bytes",
    "sampled_matching_state_bytes",
    "sampled_toehold_search_state_bytes",
    "toehold_distance_cache_bytes",
    "upper_triangle_index_batches",
]
