"""Fail-fast resource guards for every allocation-heavy search stage."""

from __future__ import annotations

from dnadesign.trijunction.errors import TriJunctionDesignError

from .estimates import (
    RequestWorkloadEstimate,
    barcode_distance_cache_bytes,
    sampled_barcode_subset_state_bytes,
    sampled_matching_state_bytes,
    toehold_distance_cache_bytes,
)
from .limits import (
    MAX_BARCODE_DISTANCE_CACHE_BYTES,
    MAX_BARCODE_DP_CELLS,
    MAX_BARCODE_GENERATION_BASE_VISITS,
    MAX_BARCODE_GENERATION_STATE_BYTES,
    MAX_BARCODE_SUBSET_LOOKUPS,
    MAX_BARCODE_SUBSET_STATE_BYTES,
    MAX_MATCHING_STATE_BYTES,
    MAX_MATCHING_SUBSTRING_VISITS,
    MAX_REQUEST_BARCODE_CANDIDATES,
    MAX_REQUEST_BARCODE_DISTANCE_CACHE_BYTES,
    MAX_REQUEST_BARCODE_DP_CELLS,
    MAX_REQUEST_BARCODE_ENCODED_BASES,
    MAX_REQUEST_BARCODE_GENERATION_BASE_VISITS,
    MAX_REQUEST_BARCODE_GENERATION_STATE_BYTES,
    MAX_REQUEST_BARCODE_SUBSET_LOOKUPS,
    MAX_REQUEST_BARCODE_SUBSET_STATE_BYTES,
    MAX_REQUEST_INPUT_BASES,
    MAX_REQUEST_LOCUS_COUNT,
    MAX_REQUEST_MATCHING_STATE_BYTES,
    MAX_REQUEST_MATCHING_SUBSTRING_VISITS,
    MAX_REQUEST_POOL_COUNT,
    MAX_REQUEST_TARGET_COUNT,
    MAX_REQUEST_TOEHOLD_CACHE_BYTES,
    MAX_REQUEST_TOEHOLD_CANDIDATES,
    MAX_REQUEST_TOEHOLD_DISTANCE_LOOKUPS,
    MAX_REQUEST_TOEHOLD_DP_CELLS,
    MAX_REQUEST_TOEHOLD_ENCODED_BASES,
    MAX_REQUEST_TOEHOLD_SEARCH_STATE_BYTES,
    MAX_TOEHOLD_CACHE_BYTES,
    MAX_TOEHOLD_DISTANCE_LOOKUPS,
    MAX_TOEHOLD_DP_CELLS,
    MAX_TOEHOLD_ENCODED_BASES,
    MAX_TOEHOLD_SEARCH_STATE_BYTES,
    REQUEST_WORKLOAD_POLICY,
)

_REQUEST_WORKLOAD_LIMITS = (
    ("physical pools", "pool_count", MAX_REQUEST_POOL_COUNT),
    ("targets", "target_count", MAX_REQUEST_TARGET_COUNT),
    ("input bases", "input_bases", MAX_REQUEST_INPUT_BASES),
    ("loci", "locus_count", MAX_REQUEST_LOCUS_COUNT),
    ("toehold candidates", "toehold_candidate_count", MAX_REQUEST_TOEHOLD_CANDIDATES),
    ("barcode candidates", "barcode_candidate_count", MAX_REQUEST_BARCODE_CANDIDATES),
    ("toehold encoded bases", "toehold_encoded_bases", MAX_REQUEST_TOEHOLD_ENCODED_BASES),
    ("toehold cache bytes", "toehold_cache_bytes", MAX_REQUEST_TOEHOLD_CACHE_BYTES),
    ("toehold distance lookups", "toehold_distance_lookups", MAX_REQUEST_TOEHOLD_DISTANCE_LOOKUPS),
    ("toehold DP cells", "toehold_dp_cells", MAX_REQUEST_TOEHOLD_DP_CELLS),
    ("toehold sampled-state bytes", "toehold_search_state_bytes", MAX_REQUEST_TOEHOLD_SEARCH_STATE_BYTES),
    (
        "barcode-generation base visits",
        "barcode_generation_base_visits",
        MAX_REQUEST_BARCODE_GENERATION_BASE_VISITS,
    ),
    (
        "barcode-generation state bytes",
        "barcode_generation_state_bytes",
        MAX_REQUEST_BARCODE_GENERATION_STATE_BYTES,
    ),
    ("barcode encoded bases", "barcode_encoded_bases", MAX_REQUEST_BARCODE_ENCODED_BASES),
    (
        "barcode-subset cache bytes",
        "barcode_distance_cache_bytes",
        MAX_REQUEST_BARCODE_DISTANCE_CACHE_BYTES,
    ),
    ("barcode-subset lookups", "barcode_subset_lookups", MAX_REQUEST_BARCODE_SUBSET_LOOKUPS),
    ("barcode-subset DP cells", "barcode_dp_cells", MAX_REQUEST_BARCODE_DP_CELLS),
    (
        "barcode-subset sampled-state bytes",
        "barcode_subset_state_bytes",
        MAX_REQUEST_BARCODE_SUBSET_STATE_BYTES,
    ),
    ("matching substring visits", "matching_substring_visits", MAX_REQUEST_MATCHING_SUBSTRING_VISITS),
    ("matching sampled-state bytes", "matching_state_bytes", MAX_REQUEST_MATCHING_STATE_BYTES),
)


def guard_request_workload(estimate: RequestWorkloadEstimate) -> None:
    """Reject unsafe aggregate work before any physical-pool search."""

    for label, field_name, limit in _REQUEST_WORKLOAD_LIMITS:
        requested = getattr(estimate, field_name)
        if requested > limit:
            raise TriJunctionDesignError(
                f"Request-wide {label} exceed the {REQUEST_WORKLOAD_POLICY} envelope: "
                f"requested {requested}, limit {limit}. Split independent physical pools across requests "
                "or lower the declared search budgets."
            )


def guard_uniform_toehold_search(
    *,
    locus_count: int,
    candidates_per_locus: int,
    sequence_length: int,
    iterations: int,
) -> None:
    """Reject unsafe uniform search shapes before candidate materialization."""

    candidate_count = locus_count * candidates_per_locus
    encoded_bases = candidate_count * sequence_length
    if encoded_bases > MAX_TOEHOLD_ENCODED_BASES:
        raise TriJunctionDesignError(
            "Toehold encoding exceeds the explicit sequence-state envelope: "
            f"{candidate_count} candidates at length {sequence_length} require {encoded_bases} bases, "
            f"limit {MAX_TOEHOLD_ENCODED_BASES}. Reduce toehold_length, or use separate pool IDs only for "
            "physically independent reactions."
        )
    required_bytes = toehold_distance_cache_bytes(candidate_count)
    if required_bytes > MAX_TOEHOLD_CACHE_BYTES:
        raise TriJunctionDesignError(
            "Toehold distance cache exceeds the explicit memory envelope: "
            f"{locus_count} loci and {candidate_count} candidates require {required_bytes} bytes, "
            f"limit {MAX_TOEHOLD_CACHE_BYTES}. Reduce search_range, or use separate pool IDs only for "
            "physically independent reactions."
        )
    requested_lookups = iterations * candidates_per_locus * locus_count * (locus_count - 1) // 2
    if requested_lookups > MAX_TOEHOLD_DISTANCE_LOOKUPS:
        raise TriJunctionDesignError(
            "Toehold search exceeds the explicit CPU envelope: "
            f"{locus_count} loci, {iterations} iterations, at most {requested_lookups} distance lookups; "
            f"limit {MAX_TOEHOLD_DISTANCE_LOOKUPS}. Lower the declared toehold_search_iterations, or use "
            "separate pool IDs only for physically independent reactions; TriJunction does not run an "
            "unbounded search."
        )
    unique_pairs = min(candidate_count * (candidate_count - 1) // 2, requested_lookups)
    dp_cells = unique_pairs * 2 * sequence_length * sequence_length
    if dp_cells > MAX_TOEHOLD_DP_CELLS:
        raise TriJunctionDesignError(
            "Toehold search exceeds the explicit edit-distance envelope: "
            f"up to {unique_pairs} unique pairs at length {sequence_length} require {dp_cells} DP cells, "
            f"limit {MAX_TOEHOLD_DP_CELLS}. Reduce toehold_length, search_range, or iterations."
        )
    sampled_state_bytes = iterations * locus_count * 12
    if sampled_state_bytes > MAX_TOEHOLD_SEARCH_STATE_BYTES:
        raise TriJunctionDesignError(
            "Toehold sampled-path state exceeds the explicit memory envelope: "
            f"{locus_count} loci and {iterations} iterations require at least {sampled_state_bytes} bytes, "
            f"limit {MAX_TOEHOLD_SEARCH_STATE_BYTES}. Lower toehold_search_iterations explicitly."
        )


def guard_barcode_generation(
    *,
    toehold_bases: int,
    length: int,
    count: int,
    forbidden_barcode_k: int,
    max_attempts: int,
) -> None:
    """Reject unsafe barcode-generation shapes before allocating pool state."""

    if count > max_attempts:
        raise TriJunctionDesignError(
            "Barcode generation cannot satisfy the declared candidate count: "
            f"{count} candidates require at least {count} attempts, but max_attempts is {max_attempts}. "
            "Increase barcode_generation_attempts or reduce barcode_pool_factor."
        )
    base_visits = max_attempts * length
    if base_visits > MAX_BARCODE_GENERATION_BASE_VISITS:
        raise TriJunctionDesignError(
            "Barcode generation exceeds the explicit CPU envelope: "
            f"{max_attempts} attempts at length {length} require {base_visits} base visits, "
            f"limit {MAX_BARCODE_GENERATION_BASE_VISITS}. Lower barcode_generation_attempts or barcode_length."
        )
    kmers_per_candidate = 2 * max(length - forbidden_barcode_k + 1, 0)
    modeled_state_bytes = count * (length + 96 * kmers_per_candidate) + toehold_bases * 96
    if modeled_state_bytes > MAX_BARCODE_GENERATION_STATE_BYTES:
        raise TriJunctionDesignError(
            "Barcode generation exceeds the explicit state envelope: "
            f"{count} candidates at length {length} require approximately {modeled_state_bytes} bytes, "
            f"limit {MAX_BARCODE_GENERATION_STATE_BYTES}. Reduce barcode_pool_factor, or use separate pool IDs "
            "only for physically independent reactions."
        )


def guard_barcode_subset_search(
    *,
    candidate_count: int,
    selected_count: int,
    sequence_length: int,
    iterations: int,
) -> None:
    """Reject unsafe subset search before sampled paths or distances exist."""

    cache_bytes = barcode_distance_cache_bytes(candidate_count)
    if cache_bytes > MAX_BARCODE_DISTANCE_CACHE_BYTES:
        raise TriJunctionDesignError(
            "Barcode distance cache exceeds the explicit memory envelope: "
            f"{candidate_count} candidates require {cache_bytes} bytes, "
            f"limit {MAX_BARCODE_DISTANCE_CACHE_BYTES}. Reduce barcode_pool_factor."
        )
    encoded_bases = candidate_count * sequence_length
    if encoded_bases > MAX_BARCODE_DISTANCE_CACHE_BYTES or sequence_length >= 65_535:
        raise TriJunctionDesignError(
            "Barcode encoding exceeds the explicit sequence-state envelope: "
            f"{candidate_count} candidates at length {sequence_length} require {encoded_bases} bases. "
            "Reduce barcode_length or barcode_pool_factor."
        )
    subset_pairs = selected_count * (selected_count - 1) // 2
    lookups = (iterations + 1) * subset_pairs
    if lookups > MAX_BARCODE_SUBSET_LOOKUPS:
        raise TriJunctionDesignError(
            "Barcode subset search exceeds the explicit CPU envelope: "
            f"{selected_count} selected barcodes and {iterations} iterations require {lookups} pair lookups, "
            f"limit {MAX_BARCODE_SUBSET_LOOKUPS}. Lower barcode_subset_iterations, or use separate pool IDs "
            "only for physically independent reactions."
        )
    unique_pairs = min(candidate_count * (candidate_count - 1) // 2, lookups)
    dp_cells = unique_pairs * sequence_length * sequence_length
    if dp_cells > MAX_BARCODE_DP_CELLS:
        raise TriJunctionDesignError(
            "Barcode subset search exceeds the explicit edit-distance envelope: "
            f"up to {unique_pairs} unique pairs at length {sequence_length} require {dp_cells} DP cells, "
            f"limit {MAX_BARCODE_DP_CELLS}. Reduce barcode_length, barcode_pool_factor, or iterations."
        )
    sampled_state_bytes = sampled_barcode_subset_state_bytes(
        evaluations=iterations + 1,
        selected_count=selected_count,
    )
    if sampled_state_bytes > MAX_BARCODE_SUBSET_STATE_BYTES:
        raise TriJunctionDesignError(
            "Barcode sampled-subset state exceeds the explicit memory envelope: "
            f"{iterations + 1} subsets of {selected_count} require at least {sampled_state_bytes} bytes, "
            f"limit {MAX_BARCODE_SUBSET_STATE_BYTES}. Lower barcode_subset_iterations."
        )


def guard_matching_search(*, count: int, combined_length: int, evaluations: int) -> None:
    """Reject unsafe final matching work and sampled state before search."""

    substring_visits = evaluations * count * combined_length * (combined_length + 1) // 2
    if substring_visits > MAX_MATCHING_SUBSTRING_VISITS:
        raise TriJunctionDesignError(
            "Toehold/barcode matching exceeds the explicit CPU envelope: "
            f"{count} junctions, {evaluations} candidate matchings, at most {substring_visits} "
            f"substring visits; limit {MAX_MATCHING_SUBSTRING_VISITS}. Lower matching_iterations, or use "
            "separate pool IDs only for physically independent reactions."
        )
    sampled_state_bytes = sampled_matching_state_bytes(evaluations=evaluations, count=count)
    if sampled_state_bytes > MAX_MATCHING_STATE_BYTES:
        raise TriJunctionDesignError(
            "Toehold/barcode sampled-matching state exceeds the explicit memory envelope: "
            f"{evaluations} matchings of {count} assignments require approximately {sampled_state_bytes} bytes, "
            f"limit {MAX_MATCHING_STATE_BYTES}. Lower matching_iterations."
        )
