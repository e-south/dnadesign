"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/design/matching.py

One-to-one toehold/barcode assignment with explicit search evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations

from dnadesign.junction.design.loci import ToeholdCandidate
from dnadesign.junction.design.randomness import StablePrng
from dnadesign.junction.design.resources import guard_matching_search


@dataclass(frozen=True, slots=True)
class JunctionAssignment:
    candidate: ToeholdCandidate
    barcode_id: str
    barcode: str


@dataclass(frozen=True, slots=True)
class MatchingSelection:
    assignments: tuple[JunctionAssignment, ...]
    matchings_evaluated: int
    max_pairwise_lcs: int


def _matching_score(
    candidates: tuple[ToeholdCandidate, ...],
    assigned_barcodes: tuple[str, ...],
) -> int:
    combined = tuple(
        candidate.sequence + barcode for candidate, barcode in zip(candidates, assigned_barcodes, strict=True)
    )
    return _maximum_shared_substring(combined)


def _maximum_shared_substring(sequences: tuple[str, ...], *, floor: int = 0) -> int:
    """Return the exact worst pairwise LCS without enumerating sequence pairs."""

    if len(sequences) < 2:
        return 0
    maximum_length = max(map(len, sequences), default=0)
    for length in range(maximum_length, floor, -1):
        seen: set[str] = set()
        for sequence in sequences:
            if len(sequence) < length:
                continue
            local = {sequence[start : start + length] for start in range(len(sequence) - length + 1)}
            if local & seen:
                return length
            seen.update(local)
    return floor


def _matching_score_with_floor(
    candidates: tuple[ToeholdCandidate, ...],
    ordered_barcodes: tuple[str, ...],
    assignment: tuple[int, ...],
    *,
    floor: int,
) -> int:
    combined = tuple(
        candidate.sequence + ordered_barcodes[barcode_index]
        for candidate, barcode_index in zip(candidates, assignment, strict=True)
    )
    return _maximum_shared_substring(combined, floor=floor)


def match_barcodes(
    candidates: tuple[ToeholdCandidate, ...],
    barcodes: tuple[str, ...],
    *,
    iterations: int,
    seed: int,
) -> MatchingSelection:
    """Minimize the worst shared substring across paired junction strings."""

    ordered_candidates = tuple(sorted(candidates, key=lambda candidate: candidate.identity))
    ordered_barcodes = tuple(sorted(barcodes))
    if len(ordered_candidates) != len(ordered_barcodes):
        raise ValueError("Toehold/barcode matching requires equal set sizes.")
    if not ordered_candidates:
        raise ValueError("Toehold/barcode matching requires at least one assignment.")
    assembly_group_ids = {candidate.assembly_group_id for candidate in ordered_candidates}
    if len(assembly_group_ids) != 1:
        raise ValueError("Toehold/barcode matching requires candidates from one assembly group.")
    assembly_group_id = next(iter(assembly_group_ids))

    count = len(ordered_barcodes)
    factorial = math.factorial(count) if count <= 8 else iterations + 1
    exhaustive = count <= 8 and factorial <= iterations
    evaluations = factorial if exhaustive else iterations + 1
    combined_length = len(ordered_candidates[0].sequence) + len(ordered_barcodes[0])
    guard_matching_search(count=count, combined_length=combined_length, evaluations=evaluations)

    if exhaustive:
        candidates_to_score = permutations(range(count))
    else:
        rng = StablePrng(seed)
        sampled: set[tuple[int, ...]] = {tuple(range(count))}
        for _ in range(iterations):
            shuffled = list(range(count))
            rng.shuffle(shuffled)
            sampled.add(tuple(shuffled))
        candidates_to_score = iter(sorted(sampled))

    fixed_floor = max(
        _maximum_shared_substring(tuple(candidate.sequence for candidate in ordered_candidates)),
        _maximum_shared_substring(ordered_barcodes),
    )
    winner: tuple[int, ...] | None = None
    winning_key: tuple[int, tuple[str, ...]] | None = None
    matchings_evaluated = 0
    for assignment in candidates_to_score:
        score = _matching_score_with_floor(
            ordered_candidates,
            ordered_barcodes,
            assignment,
            floor=fixed_floor,
        )
        assigned_barcodes = tuple(ordered_barcodes[index] for index in assignment)
        key = (score, assigned_barcodes)
        if winning_key is None or key < winning_key:
            winning_key = key
            winner = assignment
        matchings_evaluated += 1
    assert winner is not None and winning_key is not None
    score = winning_key[0]
    barcode_ids = {
        barcode: f"{assembly_group_id}:barcode-{index:04d}" for index, barcode in enumerate(ordered_barcodes, start=1)
    }
    return MatchingSelection(
        assignments=tuple(
            JunctionAssignment(
                candidate=candidate,
                barcode_id=barcode_ids[ordered_barcodes[barcode_index]],
                barcode=ordered_barcodes[barcode_index],
            )
            for candidate, barcode_index in zip(ordered_candidates, winner, strict=True)
        ),
        matchings_evaluated=matchings_evaluated,
        max_pairwise_lcs=score,
    )
