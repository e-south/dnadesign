"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/design/toeholds.py

Bounded deterministic selection of one toehold per enumerated locus.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from functools import lru_cache

import numpy as np

from dnadesign.trijunction.errors import TriJunctionDesignError
from dnadesign.trijunction.sequence import POSITION_WEIGHT_SCALE, validate_dna
from dnadesign.trijunction.sequence.distances import (
    _position_weighted_chunk_size,
    _position_weighted_levenshtein_units_encoded_many,
    _position_weighted_scratch_bytes,
)

from .loci import ToeholdCandidate, ToeholdLocus
from .randomness import StablePrng
from .resources import (
    MAX_PAIR_DISTANCE_SCRATCH_BYTES,
    MAX_TOEHOLD_CACHE_BYTES,
    guard_uniform_toehold_search,
    pair_lookup_scratch_bytes,
    pair_selection_fixed_scratch_bytes,
    pair_selection_reduction_chunk_size,
    pair_selection_reduction_scratch_bytes,
    toehold_distance_cache_bytes,
    upper_triangle_index_batches,
)
from .scoring import rank_aggregate_maximin

_EXP_WEIGHT_SCALE = 1_000_000_000_000
_SEARCH_BATCH_SIZE = 64
_UNSET_DISTANCE = np.iinfo(np.uint64).max


def _toehold_pair_scratch_bytes(pair_count: int, sequence_length: int) -> int:
    """Estimate DP plus pair-index/cache scratch for one toehold batch."""

    return _position_weighted_scratch_bytes(pair_count, sequence_length) + pair_lookup_scratch_bytes(pair_count)


def _toehold_pair_chunk_size(sequence_length: int) -> int:
    """Return one toehold pair batch inside the combined scratch budget."""

    return _position_weighted_chunk_size(
        sequence_length,
        additional_per_pair_bytes=pair_lookup_scratch_bytes(1),
        budget_bytes=MAX_PAIR_DISTANCE_SCRATCH_BYTES,
    )


def _toehold_selection_scratch_bytes(
    pair_count: int,
    *,
    sequence_length: int,
    trial_count: int,
    option_count: int,
    prior_count: int,
) -> int:
    """Estimate the complete live scratch for one selection-distance block."""

    fixed_bytes = pair_selection_fixed_scratch_bytes(
        trial_count=trial_count,
        option_count=option_count,
        prior_count=prior_count,
    )
    dynamic_bytes = max(
        _toehold_pair_scratch_bytes(pair_count, sequence_length),
        pair_selection_reduction_scratch_bytes(pair_count),
    )
    return fixed_bytes + dynamic_bytes


def _toehold_selection_pair_chunk_size(
    sequence_length: int,
    *,
    trial_count: int,
    option_count: int,
    prior_count: int,
) -> int:
    """Return a pair block size after subtracting all live selection state."""

    fixed_bytes = pair_selection_fixed_scratch_bytes(
        trial_count=trial_count,
        option_count=option_count,
        prior_count=prior_count,
    )
    available_bytes = MAX_PAIR_DISTANCE_SCRATCH_BYTES - fixed_bytes
    try:
        distance_chunk_size = _position_weighted_chunk_size(
            sequence_length,
            additional_per_pair_bytes=pair_lookup_scratch_bytes(1),
            budget_bytes=available_bytes,
        )
        reduction_chunk_size = pair_selection_reduction_chunk_size(budget_bytes=available_bytes)
    except ValueError as error:
        raise TriJunctionDesignError(
            "Toehold selection exceeds the explicit transient-memory envelope: "
            f"{trial_count} trials, {option_count} options, and {prior_count} prior choices leave "
            "insufficient space for one bounded distance pair. Reduce search_range, "
            "toehold_search_iterations, or the number of loci in the physical pool."
        ) from error
    pair_chunk_size = min(distance_chunk_size, reduction_chunk_size)
    if pair_chunk_size < trial_count:
        raise TriJunctionDesignError(
            "Toehold selection exceeds the explicit transient-memory envelope: "
            f"one option/prior block for {trial_count} trials cannot fit. Reduce toehold_length or "
            "toehold_search_iterations."
        )
    return pair_chunk_size


@dataclass(frozen=True, slots=True)
class ToeholdSelection:
    candidates: tuple[ToeholdCandidate, ...]
    minimum_distance: float
    mean_distance: float
    rank_score: float
    paths_evaluated: int


class _PairDistanceCache:
    """Lazy compact symmetric matrix for one physical pool's candidates."""

    __slots__ = ("_candidates", "_computed_pairs", "_encoded", "_size", "_values")

    def __init__(self, candidates: tuple[ToeholdCandidate, ...]) -> None:
        self._candidates = candidates
        self._size = len(candidates)
        sequence_lengths = {len(candidate.sequence) for candidate in candidates}
        if len(sequence_lengths) != 1:
            raise TriJunctionDesignError("Toehold candidates in one physical pool must have one length.")
        sequence_length = next(iter(sequence_lengths))
        sequences = tuple(validate_dna(candidate.sequence) for candidate in candidates)
        self._encoded = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8).reshape(
            self._size, sequence_length
        )
        pair_count = self._size * (self._size - 1) // 2
        required_bytes = toehold_distance_cache_bytes(self._size)
        if required_bytes > MAX_TOEHOLD_CACHE_BYTES:
            raise TriJunctionDesignError(
                "Toehold distance cache exceeds the explicit memory envelope: "
                f"{self._size} candidates require {required_bytes} bytes, limit {MAX_TOEHOLD_CACHE_BYTES}. "
                "Reduce search_range, or use separate pool IDs only for physically independent reactions."
            )
        self._values = np.full(pair_count, _UNSET_DISTANCE, dtype=np.uint64)
        self._computed_pairs = 0

    @property
    def allocated_bytes(self) -> int:
        return int(self._values.nbytes)

    @property
    def computed_pairs(self) -> int:
        return self._computed_pairs

    @property
    def pair_chunk_size(self) -> int:
        return _toehold_pair_chunk_size(self._encoded.shape[1])

    def _flat_indices(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lower = np.minimum(left, right)
        upper = np.maximum(left, right)
        return self._size * lower - lower * (lower + 1) // 2 + (upper - lower - 1)

    def distances(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return fixed-point distances for parallel candidate-index pairs."""

        if left.shape != right.shape:
            raise ValueError("candidate-index arrays must have equal shape")
        output = np.zeros(left.shape, dtype=np.uint64)
        if output.size == 0:
            return output

        output_flat = output.reshape(-1)
        left_flat = left.reshape(-1) if left.flags.c_contiguous else None
        right_flat = right.reshape(-1) if right.flags.c_contiguous else None
        pair_chunk_size = self.pair_chunk_size
        for start in range(0, output.size, pair_chunk_size):
            stop = min(start + pair_chunk_size, output.size)
            output_chunk = output_flat[start:stop]
            chunk_left = left_flat[start:stop] if left_flat is not None else left.flat[start:stop]
            chunk_right = right_flat[start:stop] if right_flat is not None else right.flat[start:stop]
            distinct_mask = chunk_left != chunk_right
            if not np.any(distinct_mask):
                continue
            distinct_left = chunk_left[distinct_mask].astype(np.int64, copy=False)
            distinct_right = chunk_right[distinct_mask].astype(np.int64, copy=False)
            flat_indices = self._flat_indices(distinct_left, distinct_right)
            missing_positions = np.flatnonzero(self._values[flat_indices] == _UNSET_DISTANCE)
            if missing_positions.size:
                missing_flat = flat_indices[missing_positions]
                still_missing = self._values[missing_flat] == _UNSET_DISTANCE
                positions = missing_positions[still_missing]
                missing_flat = missing_flat[still_missing]
                if positions.size:
                    unique_flat, first_offsets = np.unique(missing_flat, return_index=True)
                    first_positions = positions[first_offsets]
                    values = _position_weighted_levenshtein_units_encoded_many(
                        self._encoded[distinct_left[first_positions]],
                        self._encoded[distinct_right[first_positions]],
                    )
                    self._values[unique_flat] = values
                    self._computed_pairs += len(unique_flat)
            output_chunk[distinct_mask] = self._values[flat_indices]
        return output


@lru_cache(maxsize=65_536)
def _stable_exp_weight(delta_units: int) -> int:
    """Return a fixed integer approximation to ``exp(delta)`` for delta <= 0."""

    if delta_units > 0:
        raise ValueError("softmax deltas must be nonpositive")
    with localcontext() as context:
        context.prec = 50
        exponent = Decimal(delta_units) / Decimal(POSITION_WEIGHT_SCALE)
        scaled = exponent.exp() * Decimal(_EXP_WEIGHT_SCALE)
        return int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))


def _selection_pair_blocks(
    options: np.ndarray,
    previous: np.ndarray,
    *,
    pair_chunk_size: int,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray, tuple[int, int, int]]]:
    """Materialize bounded option/prior blocks in deterministic C order."""

    trial_count, option_count = options.shape
    prior_count = previous.shape[1]
    pairs_per_option_prior = trial_count
    cells_per_trial = pair_chunk_size // pairs_per_option_prior
    if cells_per_trial < 1:
        raise TriJunctionDesignError("Toehold selection cannot fit one trial block in its memory envelope.")

    option_chunk_size = min(option_count, cells_per_trial)
    for option_start in range(0, option_count, option_chunk_size):
        option_stop = min(option_start + option_chunk_size, option_count)
        block_option_count = option_stop - option_start
        prior_chunk_size = max(1, cells_per_trial // block_option_count)
        for prior_start in range(0, prior_count, prior_chunk_size):
            prior_stop = min(prior_start + prior_chunk_size, prior_count)
            block_prior_count = prior_stop - prior_start
            shape = (trial_count, block_option_count, block_prior_count)
            left = np.ascontiguousarray(np.broadcast_to(options[:, option_start:option_stop, None], shape)).reshape(-1)
            right = np.ascontiguousarray(np.broadcast_to(previous[:, None, prior_start:prior_stop], shape)).reshape(-1)
            yield option_start, option_stop, left, right, shape
            del left, right


def _selection_weights_streamed(
    options: np.ndarray,
    previous: np.ndarray,
    distances: _PairDistanceCache,
    *,
    pair_chunk_size: int,
) -> np.ndarray:
    """Return exact fixed-point choice weights without a full pair tensor."""

    if options.ndim != 2 or previous.ndim != 2 or options.shape[0] != previous.shape[0]:
        raise ValueError("selection option and prior-choice matrices must share one trial axis")
    trial_count, option_count = options.shape
    maxima = np.zeros(trial_count, dtype=np.uint64)
    for _option_start, _option_stop, left, right, shape in _selection_pair_blocks(
        options,
        previous,
        pair_chunk_size=pair_chunk_size,
    ):
        block_maxima = distances.distances(left, right).reshape(shape).max(axis=(1, 2))
        np.maximum(maxima, block_maxima, out=maxima)
        del left, right, block_maxima

    maxima_signed = maxima.astype(np.int64)
    weights = np.zeros((trial_count, option_count), dtype=np.uint64)
    for option_start, option_stop, left, right, shape in _selection_pair_blocks(
        options,
        previous,
        pair_chunk_size=pair_chunk_size,
    ):
        block_distances = distances.distances(left, right).reshape(shape)
        deltas = block_distances.astype(np.int64)
        deltas -= maxima_signed[:, None, None]
        distinct, inverse = np.unique(deltas, return_inverse=True)
        lookup = np.asarray(
            [_stable_exp_weight(int(delta)) for delta in distinct],
            dtype=np.uint64,
        )
        contributions = lookup[inverse].reshape(shape)
        weights[:, option_start:option_stop] += contributions.sum(axis=2, dtype=np.uint64)
        del block_distances, contributions, deltas, distinct, inverse, left, lookup, right

    baselines = np.asarray([_stable_exp_weight(-int(maximum)) for maximum in maxima], dtype=np.uint64)
    weights += baselines[:, None]
    return weights


def _path_score(path_indices: tuple[int, ...], distances: _PairDistanceCache) -> tuple[int, Fraction]:
    if len(path_indices) < 2:
        return (0, Fraction(0))
    minimum: int | None = None
    total = 0
    pair_count = 0
    for left, right in upper_triangle_index_batches(path_indices, batch_size=distances.pair_chunk_size):
        values = distances.distances(left, right)
        batch_minimum = int(values.min())
        minimum = batch_minimum if minimum is None else min(minimum, batch_minimum)
        total += sum(int(value) for value in values)
        pair_count += len(values)
    assert minimum is not None
    return (minimum, Fraction(total, pair_count))


def select_toeholds(
    loci: tuple[ToeholdLocus, ...],
    *,
    iterations: int,
    seed: int,
) -> ToeholdSelection:
    """Choose one candidate per locus with a seeded bounded maximin search."""

    ordered_loci = tuple(sorted(loci, key=lambda locus: locus.identity))
    if not ordered_loci:
        raise TriJunctionDesignError("A physical pool has no toehold loci to design.")
    candidate_counts = tuple(len(locus.candidates) for locus in ordered_loci)
    if len(set(candidate_counts)) != 1:
        raise TriJunctionDesignError("Every toehold locus in one physical pool must have one candidate count.")
    guard_uniform_toehold_search(
        locus_count=len(ordered_loci),
        candidates_per_locus=candidate_counts[0],
        sequence_length=len(ordered_loci[0].candidates[0].sequence),
        iterations=iterations,
    )
    selection_pair_chunk_size: int | None = None
    if len(ordered_loci) > 1:
        selection_pair_chunk_size = _toehold_selection_pair_chunk_size(
            len(ordered_loci[0].candidates[0].sequence),
            trial_count=min(iterations, _SEARCH_BATCH_SIZE),
            option_count=candidate_counts[0],
            prior_count=len(ordered_loci) - 1,
        )

    all_candidates = tuple(
        candidate for locus in ordered_loci for candidate in sorted(locus.candidates, key=lambda item: item.identity)
    )
    index_by_identity = {candidate.identity: index for index, candidate in enumerate(all_candidates)}
    locus_indices = tuple(
        tuple(
            index_by_identity[candidate.identity]
            for candidate in sorted(locus.candidates, key=lambda item: item.identity)
        )
        for locus in ordered_loci
    )
    distance_cache = _PairDistanceCache(all_candidates)
    locus_matrix = np.asarray(locus_indices, dtype=np.int64)
    locus_count, candidate_count = locus_matrix.shape
    seeder = StablePrng(seed)
    trial_rngs = [StablePrng(seeder.next_u64()) for _ in range(iterations)]
    visit_orders = np.empty((iterations, locus_count), dtype=np.int32)
    selected = np.empty((iterations, locus_count), dtype=np.int32)
    for trial_index, rng in enumerate(trial_rngs):
        order = list(range(locus_count))
        rng.shuffle(order)
        visit_orders[trial_index] = order
        first_options = locus_matrix[order[0]]
        selected[trial_index, 0] = first_options[rng.randbelow(candidate_count)]

    for step in range(1, locus_count):
        for batch_start in range(0, iterations, _SEARCH_BATCH_SIZE):
            batch_end = min(batch_start + _SEARCH_BATCH_SIZE, iterations)
            rows = np.arange(batch_start, batch_end)
            options = locus_matrix[visit_orders[rows, step]]
            previous = selected[rows, :step].astype(np.int64, copy=False)
            assert selection_pair_chunk_size is not None
            weights = _selection_weights_streamed(
                options,
                previous,
                distance_cache,
                pair_chunk_size=selection_pair_chunk_size,
            )
            for offset, trial_index in enumerate(rows):
                choice = trial_rngs[int(trial_index)].weighted_choice(
                    range(candidate_count),
                    tuple(int(weight) for weight in weights[offset]),
                )
                selected[int(trial_index), step] = options[offset, choice]

    canonical_paths = np.empty_like(selected)
    for trial_index in range(iterations):
        canonical_paths[trial_index, visit_orders[trial_index]] = selected[trial_index]
    baseline = locus_matrix[:, 0].astype(np.int32, copy=False)
    unique_paths = np.unique(np.vstack((baseline[None, :], canonical_paths)), axis=0)
    paths_by_identity: dict[tuple[tuple[str, int, int], ...], tuple[int, ...]] = {}
    for path_array in unique_paths:
        path = tuple(int(index) for index in path_array)
        identity = tuple(all_candidates[index].identity for index in path)
        paths_by_identity[identity] = path

    scores = {identity: _path_score(path, distance_cache) for identity, path in paths_by_identity.items()}
    ranks = rank_aggregate_maximin(scores)
    winning_rank = max(rank.weighted_score_fraction for rank in ranks.values())
    winning_identity = min(identity for identity, rank in ranks.items() if rank.weighted_score_fraction == winning_rank)
    winner_indices = paths_by_identity[winning_identity]
    score = scores[winning_identity]
    return ToeholdSelection(
        candidates=tuple(all_candidates[index] for index in winner_indices),
        minimum_distance=score[0] / POSITION_WEIGHT_SCALE,
        mean_distance=float(score[1] / POSITION_WEIGHT_SCALE),
        rank_score=ranks[winning_identity].weighted_score,
        paths_evaluated=len(scores),
    )


__all__ = ["ToeholdSelection", "select_toeholds"]
