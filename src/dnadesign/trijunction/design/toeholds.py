"""Bounded deterministic selection of one toehold per enumerated locus."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from functools import lru_cache

import numpy as np

from dnadesign.trijunction.errors import TriJunctionDesignError
from dnadesign.trijunction.sequence import POSITION_WEIGHT_SCALE, validate_dna
from dnadesign.trijunction.sequence.distances import _position_weighted_levenshtein_units_encoded_many

from .loci import ToeholdCandidate, ToeholdLocus
from .randomness import StablePrng
from .resources import (
    MAX_TOEHOLD_CACHE_BYTES,
    guard_uniform_toehold_search,
    toehold_distance_cache_bytes,
)
from .scoring import rank_aggregate_maximin

_EXP_WEIGHT_SCALE = 1_000_000_000_000
_SEARCH_BATCH_SIZE = 64
_UNSET_DISTANCE = np.iinfo(np.uint64).max


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

    def _flat_indices(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lower = np.minimum(left, right)
        upper = np.maximum(left, right)
        return self._size * lower - lower * (lower + 1) // 2 + (upper - lower - 1)

    def distances(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return fixed-point distances for parallel candidate-index pairs."""

        if left.shape != right.shape:
            raise ValueError("candidate-index arrays must have equal shape")
        output = np.zeros(left.shape, dtype=np.uint64)
        distinct_mask = left != right
        if not np.any(distinct_mask):
            return output

        distinct_left = left[distinct_mask].astype(np.int64, copy=False)
        distinct_right = right[distinct_mask].astype(np.int64, copy=False)
        flat_indices = self._flat_indices(distinct_left, distinct_right)
        missing_positions = np.flatnonzero(self._values[flat_indices] == _UNSET_DISTANCE)
        if missing_positions.size:
            missing_flat = flat_indices[missing_positions]
            unique_flat, first_offsets = np.unique(missing_flat, return_index=True)
            first_positions = missing_positions[first_offsets]
            values = _position_weighted_levenshtein_units_encoded_many(
                self._encoded[distinct_left[first_positions]],
                self._encoded[distinct_right[first_positions]],
            )
            self._values[unique_flat] = values
            self._computed_pairs += len(unique_flat)
        output[distinct_mask] = self._values[flat_indices]
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


def _selection_weights_batch(distances: np.ndarray) -> np.ndarray:
    """Return integer weights for ``trial x option x prior-choice`` distances."""

    maxima = distances.max(axis=(1, 2))
    deltas = distances.astype(np.int64) - maxima[:, None, None].astype(np.int64)
    distinct, inverse = np.unique(deltas, return_inverse=True)
    lookup = np.asarray(
        [_stable_exp_weight(int(delta)) for delta in distinct],
        dtype=np.uint64,
    )
    contributions = lookup[inverse].reshape(distances.shape)
    baselines = np.asarray([_stable_exp_weight(-int(maximum)) for maximum in maxima], dtype=np.uint64)
    return contributions.sum(axis=2, dtype=np.uint64) + baselines[:, None]


def _path_score(path_indices: tuple[int, ...], distances: _PairDistanceCache) -> tuple[int, Fraction]:
    if len(path_indices) < 2:
        return (0, Fraction(0))
    row, column = np.triu_indices(len(path_indices), 1)
    indices = np.asarray(path_indices, dtype=np.int64)
    values = distances.distances(indices[row], indices[column])
    total = sum(int(value) for value in values)
    return (int(values.min()), Fraction(total, len(values)))


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
            batch_size = len(rows)
            distances = distance_cache.distances(
                np.broadcast_to(options[:, :, None], (batch_size, candidate_count, step)).reshape(-1),
                np.broadcast_to(previous[:, None, :], (batch_size, candidate_count, step)).reshape(-1),
            ).reshape(batch_size, candidate_count, step)
            weights = _selection_weights_batch(distances)
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
