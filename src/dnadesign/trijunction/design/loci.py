"""Versioned junction-locus geometry and complete candidate enumeration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dnadesign.trijunction.errors import TriJunctionDesignError

from .resources import guard_uniform_toehold_search

if TYPE_CHECKING:
    from dnadesign.trijunction.contracts.request import PlanningProfile, Target


@dataclass(frozen=True, slots=True)
class ToeholdCandidate:
    target_id: str
    pool_id: str
    locus_index: int
    candidate_offset: int
    start: int
    sequence: str

    @property
    def identity(self) -> tuple[str, int, int]:
        return (self.target_id, self.locus_index, self.candidate_offset)


@dataclass(frozen=True, slots=True)
class ToeholdLocus:
    target_id: str
    pool_id: str
    index: int
    candidates: tuple[ToeholdCandidate, ...]

    @property
    def identity(self) -> tuple[str, int]:
        return (self.target_id, self.index)


def predict_locus_count(target_length: int, profile: PlanningProfile) -> int:
    """Predict the v1 order-ceiling-safe locus count without allocating candidates."""

    fragment_stride = profile.oligo_length - 2 * profile.barcode_length
    trailing_domain_limit = profile.oligo_length - profile.barcode_length
    locus_start = profile.oligo_length - profile.barcode_length - profile.toehold_length
    first_candidate_end = locus_start + profile.search_range - 1 + profile.toehold_length
    if first_candidate_end > target_length:
        return 0
    first_trailing_domain = target_length - (locus_start + profile.toehold_length)
    excess_trailing_domain = max(0, first_trailing_domain - trailing_domain_limit)
    additional_loci = (excess_trailing_domain + fragment_stride - 1) // fragment_stride
    return 1 + additional_loci


def enumerate_loci(target: Target, profile: PlanningProfile) -> tuple[ToeholdLocus, ...]:
    """Enumerate complete candidate windows for one exact target."""

    fragment_stride = profile.oligo_length - 2 * profile.barcode_length
    trailing_domain_limit = profile.oligo_length - profile.barcode_length
    locus_start = profile.oligo_length - profile.barcode_length - profile.toehold_length
    last_candidate_end = locus_start + profile.search_range - 1 + profile.toehold_length
    if last_candidate_end > len(target.sequence):
        return ()

    predicted_locus_count = predict_locus_count(len(target.sequence), profile)
    guard_uniform_toehold_search(
        locus_count=predicted_locus_count,
        candidates_per_locus=profile.search_range,
        sequence_length=profile.toehold_length,
        iterations=profile.toehold_search_iterations,
    )
    loci: list[ToeholdLocus] = []
    while True:
        candidates: list[ToeholdCandidate] = []
        for offset in range(profile.search_range):
            start = locus_start + offset
            sequence = target.sequence[start : start + profile.toehold_length]
            if len(sequence) != profile.toehold_length:
                raise TriJunctionDesignError(
                    f"Target '{target.id}' cannot provide {profile.search_range} complete candidates "
                    f"at locus {len(loci)}; candidate offset {offset} is truncated."
                )
            candidates.append(
                ToeholdCandidate(
                    target_id=target.id,
                    pool_id=target.pool_id,
                    locus_index=len(loci),
                    candidate_offset=offset,
                    start=start,
                    sequence=sequence,
                )
            )
        loci.append(
            ToeholdLocus(
                target_id=target.id,
                pool_id=target.pool_id,
                index=len(loci),
                candidates=tuple(candidates),
            )
        )
        trailing_domain_length = len(target.sequence) - (locus_start + profile.toehold_length)
        if trailing_domain_length <= trailing_domain_limit:
            break
        locus_start += fragment_stride

    if len(loci) != predicted_locus_count:
        raise AssertionError("predicted locus count diverged from enumeration")
    return tuple(loci)


__all__ = ["ToeholdCandidate", "ToeholdLocus", "enumerate_loci", "predict_locus_count"]
