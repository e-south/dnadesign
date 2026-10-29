"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/design/barcodes.py

Deterministic barcode-pool generation and maximin subset selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from dnadesign.junction.errors import JunctionDesignError
from dnadesign.junction.sequence import (
    kmer_set,
    kmer_set_with_reverse_complements,
    reverse_complement,
    validate_dna,
)
from dnadesign.junction.sequence.distances import (
    _levenshtein_chunk_size,
    _levenshtein_distance_encoded_many,
    _levenshtein_scratch_bytes,
)

from .randomness import StablePrng
from .resources import (
    MAX_PAIR_DISTANCE_SCRATCH_BYTES,
    guard_barcode_generation,
    guard_barcode_subset_search,
    pair_lookup_scratch_bytes,
    upper_triangle_index_batches,
)
from .scoring import rank_aggregate_maximin

# This ordering is part of the v1 seeded-search contract, not biological order.
_DNA = "AGTC"
_UNSET_BARCODE_DISTANCE = np.iinfo(np.uint16).max


def _barcode_pair_scratch_bytes(pair_count: int, sequence_length: int) -> int:
    """Estimate DP plus pair-index/cache scratch for one barcode batch."""

    return _levenshtein_scratch_bytes(pair_count, sequence_length) + pair_lookup_scratch_bytes(pair_count)


def _barcode_pair_chunk_size(sequence_length: int) -> int:
    """Return one barcode pair batch inside the combined scratch budget."""

    return _levenshtein_chunk_size(
        sequence_length,
        additional_per_pair_bytes=pair_lookup_scratch_bytes(1),
        budget_bytes=MAX_PAIR_DISTANCE_SCRATCH_BYTES,
    )


@dataclass(frozen=True, slots=True)
class BarcodeSelection:
    barcodes: tuple[str, ...]
    candidates_generated: int
    forbidden_toehold_k: int
    forbidden_barcode_k: int
    subsets_evaluated: int
    minimum_distance: float
    mean_distance: float
    rank_score: float


def _gc_fraction(sequence: str) -> float:
    return sum(base in {"G", "C"} for base in sequence) / len(sequence)


def _maximum_homopolymer(sequence: str) -> int:
    longest = 1
    current = 1
    for previous, base in zip(sequence, sequence[1:], strict=False):
        current = current + 1 if base == previous else 1
        longest = max(longest, current)
    return longest


def _random_dna(rng: StablePrng, length: int) -> str:
    """Draw DNA from stable two-bit chunks without per-base helper drift."""

    bases: list[str] = []
    while len(bases) < length:
        draw = rng.next_u64()
        for offset in range(0, 64, 2):
            bases.append(_DNA[(draw >> offset) & 0b11])
            if len(bases) == length:
                break
    return "".join(bases)


def _sample_candidate_indices(
    rng: StablePrng,
    *,
    population_size: int,
    selected_count: int,
) -> tuple[int, ...]:
    """Reproduce partial Fisher-Yates sampling with selected-count state."""

    swaps: dict[int, int] = {}
    selected: list[int] = []
    for index in range(selected_count):
        swap_index = index + rng.randbelow(population_size - index)
        index_value = swaps.pop(index, index)
        if swap_index == index:
            selected.append(index_value)
            continue
        selected.append(swaps.pop(swap_index, swap_index))
        swaps[swap_index] = index_value
    return tuple(selected)


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
    """Generate an explicit SSM-like pool without silent relaxation."""

    guard_barcode_generation(
        toehold_count=len(toeholds),
        toehold_length=max(map(len, toeholds), default=0),
        length=length,
        count=count,
        forbidden_toehold_k=forbidden_toehold_k,
        forbidden_barcode_k=forbidden_barcode_k,
        max_attempts=max_attempts,
    )
    rng = StablePrng(seed)
    forbidden_toehold_substrings: set[str] = set()
    for toehold in toeholds:
        forbidden_toehold_substrings.update(kmer_set_with_reverse_complements(toehold, forbidden_toehold_k))

    accepted: list[str] = []
    accepted_sequences: set[str] = set()
    accepted_barcode_substrings: set[str] = set()
    for _ in range(max_attempts):
        candidate = _random_dna(rng, length)
        candidate_rc = reverse_complement(candidate)
        if candidate in accepted_sequences or candidate_rc in accepted_sequences:
            continue
        gc_fraction = _gc_fraction(candidate)
        if not gc_min <= gc_fraction <= gc_max:
            continue
        if _maximum_homopolymer(candidate) > max_homopolymer:
            continue
        if kmer_set_with_reverse_complements(candidate, forbidden_toehold_k) & forbidden_toehold_substrings:
            continue

        forward_kmers = kmer_set(candidate, forbidden_barcode_k)
        reverse_kmers = kmer_set(candidate_rc, forbidden_barcode_k)
        if forward_kmers & reverse_kmers:
            continue
        candidate_kmers = forward_kmers | reverse_kmers
        if candidate_kmers & accepted_barcode_substrings:
            continue

        accepted.append(candidate)
        accepted_sequences.add(candidate)
        accepted_barcode_substrings.update(candidate_kmers)
        if len(accepted) == count:
            return tuple(accepted)

    raise JunctionDesignError(
        "Barcode generation exhausted its explicit attempt budget: "
        f"needed {count}, generated {len(accepted)}, attempts {max_attempts}, "
        f"toehold k={forbidden_toehold_k}, barcode k={forbidden_barcode_k}. "
        "Change the planning profile explicitly; junction does not relax constraints silently."
    )


class _BarcodeDistanceCache:
    """Lazy compact unit-edit matrix for one generated candidate pool."""

    __slots__ = ("_encoded", "_size", "_values")

    def __init__(self, candidates: tuple[str, ...]) -> None:
        sequence_lengths = {len(candidate) for candidate in candidates}
        if len(sequence_lengths) != 1:
            raise JunctionDesignError("Barcode candidates must have one common length.")
        if len(set(candidates)) != len(candidates):
            raise JunctionDesignError("Barcode candidates must be unique.")
        sequence_length = next(iter(sequence_lengths))
        sequences = tuple(validate_dna(candidate) for candidate in candidates)
        self._size = len(sequences)
        self._encoded = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8).reshape(
            self._size, sequence_length
        )
        self._values = np.full(
            self._size * (self._size - 1) // 2,
            _UNSET_BARCODE_DISTANCE,
            dtype=np.uint16,
        )

    def _flat_indices(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lower = np.minimum(left, right)
        upper = np.maximum(left, right)
        return self._size * lower - lower * (lower + 1) // 2 + (upper - lower - 1)

    @property
    def pair_chunk_size(self) -> int:
        return _barcode_pair_chunk_size(self._encoded.shape[1])

    def distances(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if left.shape != right.shape:
            raise ValueError("barcode-index arrays must have equal shape")
        output = np.zeros(left.shape, dtype=np.uint16)
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
            missing_positions = np.flatnonzero(self._values[flat_indices] == _UNSET_BARCODE_DISTANCE)
            if missing_positions.size:
                missing_flat = flat_indices[missing_positions]
                still_missing = self._values[missing_flat] == _UNSET_BARCODE_DISTANCE
                positions = missing_positions[still_missing]
                missing_flat = missing_flat[still_missing]
                if positions.size:
                    unique_flat, first_offsets = np.unique(missing_flat, return_index=True)
                    first_positions = positions[first_offsets]
                    values = _levenshtein_distance_encoded_many(
                        self._encoded[distinct_left[first_positions]],
                        self._encoded[distinct_right[first_positions]],
                    )
                    self._values[unique_flat] = values.astype(np.uint16)
            output_chunk[distinct_mask] = self._values[flat_indices]
        return output


def _subset_score_indices(indices: tuple[int, ...], cache: _BarcodeDistanceCache) -> tuple[int, Fraction]:
    if len(indices) < 2:
        return (0, Fraction(0))
    minimum: int | None = None
    total = 0
    pair_count = 0
    for left, right in upper_triangle_index_batches(indices, batch_size=cache.pair_chunk_size):
        distances = cache.distances(left, right)
        batch_minimum = int(distances.min())
        minimum = batch_minimum if minimum is None else min(minimum, batch_minimum)
        total += sum(int(distance) for distance in distances)
        pair_count += len(distances)
    assert minimum is not None
    return (minimum, Fraction(total, pair_count))


def select_barcodes(
    candidates: tuple[str, ...],
    *,
    count: int,
    iterations: int,
    seed: int,
    forbidden_toehold_k: int,
    forbidden_barcode_k: int,
) -> BarcodeSelection:
    """Choose a seeded barcode subset, prioritizing worst-pair distance."""

    if len(candidates) < count:
        raise JunctionDesignError(f"Barcode pool has {len(candidates)} candidates but {count} are required.")
    sequence_lengths = {len(candidate) for candidate in candidates}
    if len(sequence_lengths) != 1:
        raise JunctionDesignError("Barcode candidates must have one common length.")
    guard_barcode_subset_search(
        candidate_count=len(candidates),
        selected_count=count,
        sequence_length=next(iter(sequence_lengths)),
        iterations=iterations,
    )
    rng = StablePrng(seed)
    sampled: set[tuple[int, ...]] = {tuple(range(count))}
    for _ in range(iterations):
        sampled.add(
            tuple(
                sorted(
                    _sample_candidate_indices(
                        rng,
                        population_size=len(candidates),
                        selected_count=count,
                    )
                )
            )
        )

    cache = _BarcodeDistanceCache(candidates)
    scores = {subset: _subset_score_indices(subset, cache) for subset in sampled}
    ranks = rank_aggregate_maximin(scores)
    winning_rank = max(rank.weighted_score_fraction for rank in ranks.values())
    winner = min(subset for subset, rank in ranks.items() if rank.weighted_score_fraction == winning_rank)
    score = scores[winner]
    return BarcodeSelection(
        barcodes=tuple(sorted(candidates[index] for index in winner)),
        candidates_generated=len(candidates),
        forbidden_toehold_k=forbidden_toehold_k,
        forbidden_barcode_k=forbidden_barcode_k,
        subsets_evaluated=len(scores),
        minimum_distance=float(score[0]),
        mean_distance=float(score[1]),
        rank_score=ranks[winner].weighted_score,
    )
