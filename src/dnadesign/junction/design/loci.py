"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/design/loci.py

Versioned junction-locus geometry and complete candidate enumeration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dnadesign.junction.errors import JunctionDesignError

from .resources import guard_uniform_toehold_search

if TYPE_CHECKING:
    from dnadesign.junction.contracts.request import PlanningProfile, Target


@dataclass(frozen=True, slots=True)
class ToeholdCandidate:
    target_id: str
    assembly_group_id: str
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
    assembly_group_id: str
    index: int
    candidates: tuple[ToeholdCandidate, ...]

    @property
    def identity(self) -> tuple[str, int]:
        return (self.target_id, self.index)


def predict_locus_count(target_length: int, profile: PlanningProfile) -> int:
    """Predict the v1 order-ceiling-safe locus count without allocating candidates."""

    nominal_length = profile.nominal_fragment_oligo_length
    fragment_stride = nominal_length - 2 * profile.barcode_length
    trailing_domain_limit = nominal_length - profile.barcode_length
    locus_start = nominal_length - profile.barcode_length - profile.toehold_length
    first_candidate_end = locus_start + profile.search_range - 1 + profile.toehold_length
    if first_candidate_end >= target_length:
        return 0
    first_trailing_domain = target_length - (locus_start + profile.toehold_length)
    excess_trailing_domain = max(0, first_trailing_domain - trailing_domain_limit)
    additional_loci = (excess_trailing_domain + fragment_stride - 1) // fragment_stride
    locus_count = 1 + additional_loci
    final_locus_start = locus_start + additional_loci * fragment_stride
    final_candidate_end = final_locus_start + profile.search_range - 1 + profile.toehold_length
    if final_candidate_end >= target_length:
        return 0
    return locus_count


def enumerate_loci(target: Target, profile: PlanningProfile) -> tuple[ToeholdLocus, ...]:
    """Enumerate complete candidate windows for one exact target."""

    nominal_length = profile.nominal_fragment_oligo_length
    fragment_stride = nominal_length - 2 * profile.barcode_length
    trailing_domain_limit = nominal_length - profile.barcode_length
    locus_start = nominal_length - profile.barcode_length - profile.toehold_length
    last_candidate_end = locus_start + profile.search_range - 1 + profile.toehold_length
    if last_candidate_end >= len(target.sequence):
        return ()

    predicted_locus_count = predict_locus_count(len(target.sequence), profile)
    if predicted_locus_count == 0:
        return ()
    guard_uniform_toehold_search(
        locus_count=predicted_locus_count,
        candidates_per_locus=profile.search_range,
        sequence_length=profile.toehold_length,
        iterations=profile.toehold_search_iterations,
    )
    loci: list[ToeholdLocus] = []
    while True:
        if locus_start + profile.search_range - 1 + profile.toehold_length >= len(target.sequence):
            raise JunctionDesignError(
                f"Target '{target.id}' cannot provide a complete candidate locus with a nonempty terminal domain."
            )
        candidates: list[ToeholdCandidate] = []
        for offset in range(profile.search_range):
            start = locus_start + offset
            sequence = target.sequence[start : start + profile.toehold_length]
            if len(sequence) != profile.toehold_length:
                raise JunctionDesignError(
                    f"Target '{target.id}' cannot provide {profile.search_range} complete candidates "
                    f"at locus {len(loci)}; candidate offset {offset} is truncated."
                )
            candidates.append(
                ToeholdCandidate(
                    target_id=target.id,
                    assembly_group_id=target.assembly_group_id,
                    locus_index=len(loci),
                    candidate_offset=offset,
                    start=start,
                    sequence=sequence,
                )
            )
        loci.append(
            ToeholdLocus(
                target_id=target.id,
                assembly_group_id=target.assembly_group_id,
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


__all__ = [
    "ToeholdCandidate",
    "ToeholdLocus",
    "enumerate_loci",
    "predict_locus_count",
]
