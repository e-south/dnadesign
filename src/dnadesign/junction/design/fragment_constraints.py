"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/design/fragment_constraints.py

Candidate-path constraints derived from physical fragment-order lengths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.junction.design.loci import ToeholdCandidate, ToeholdLocus
from dnadesign.junction.errors import JunctionDesignError


@dataclass(frozen=True, slots=True)
class FragmentPathConstraint:
    """Exact fragment-order floor over one assembly group's candidate paths."""

    target_lengths: tuple[tuple[str, int], ...]
    barcode_length: int
    toehold_length: int
    minimum_fragment_oligo_length: int

    def _last_fragment_is_feasible(self, candidate: ToeholdCandidate, *, target_length: int) -> bool:
        terminal_domain_length = target_length - (candidate.start + self.toehold_length)
        barcode_bearing_length = self.barcode_length + terminal_domain_length
        complement_length = terminal_domain_length + self.toehold_length
        return min(barcode_bearing_length, complement_length) >= self.minimum_fragment_oligo_length

    def allows(self, candidates: tuple[ToeholdCandidate, ...]) -> bool:
        """Return whether every emitted fragment order meets the declared floor."""

        by_target: dict[str, list[ToeholdCandidate]] = {}
        for candidate in candidates:
            by_target.setdefault(candidate.target_id, []).append(candidate)
        target_lengths = dict(self.target_lengths)
        if set(by_target) != set(target_lengths):
            return False
        for target_id in sorted(by_target):
            target_length = target_lengths.get(target_id)
            if target_length is None:
                raise JunctionDesignError(f"Fragment path constraint has no target length for {target_id!r}.")
            selected = tuple(sorted(by_target[target_id], key=lambda candidate: candidate.locus_index))
            if tuple(candidate.locus_index for candidate in selected) != tuple(range(len(selected))):
                return False
            if selected[0].start < self.minimum_fragment_oligo_length:
                return False
            if any(
                current.start - previous.start < self.minimum_fragment_oligo_length
                for previous, current in zip(selected, selected[1:], strict=False)
            ):
                return False
            if not self._last_fragment_is_feasible(selected[-1], target_length=target_length):
                return False
        return True

    def first_feasible_path(self, loci: tuple[ToeholdLocus, ...]) -> tuple[ToeholdCandidate, ...]:
        """Return the lexical first feasible path without enumerating path products."""

        by_target: dict[str, list[ToeholdLocus]] = {}
        for locus in sorted(loci, key=lambda item: item.identity):
            by_target.setdefault(locus.target_id, []).append(locus)
        target_lengths = dict(self.target_lengths)

        selected_by_identity: dict[tuple[str, int], ToeholdCandidate] = {}
        for target_id in sorted(by_target):
            target_loci = tuple(by_target[target_id])
            target_length = target_lengths.get(target_id)
            if target_length is None:
                raise JunctionDesignError(f"Fragment path constraint has no target length for {target_id!r}.")
            maximum_feasible_starts = [0] * len(target_loci)
            maximum_last_start = max(
                (
                    candidate.start
                    for candidate in target_loci[-1].candidates
                    if self._last_fragment_is_feasible(candidate, target_length=target_length)
                ),
                default=None,
            )
            if maximum_last_start is None:
                self._raise_no_path(target_id)
            maximum_feasible_starts[-1] = maximum_last_start
            for index in range(len(target_loci) - 2, -1, -1):
                maximum_next_start = maximum_feasible_starts[index + 1]
                maximum_current_start = max(
                    (
                        candidate.start
                        for candidate in target_loci[index].candidates
                        if candidate.start + self.minimum_fragment_oligo_length <= maximum_next_start
                    ),
                    default=None,
                )
                if maximum_current_start is None:
                    self._raise_no_path(target_id)
                maximum_feasible_starts[index] = maximum_current_start

            previous: ToeholdCandidate | None = None
            for index, locus in enumerate(target_loci):
                lower_bound = (
                    self.minimum_fragment_oligo_length
                    if previous is None
                    else previous.start + self.minimum_fragment_oligo_length
                )
                candidate = min(
                    (item for item in locus.candidates if lower_bound <= item.start <= maximum_feasible_starts[index]),
                    key=lambda value: value.identity,
                    default=None,
                )
                if candidate is None:
                    self._raise_no_path(target_id)
                selected_by_identity[locus.identity] = candidate
                previous = candidate

        path = tuple(selected_by_identity[locus.identity] for locus in sorted(loci, key=lambda item: item.identity))
        if not self.allows(path):
            raise AssertionError("fragment path feasibility construction diverged from validation")
        return path

    def _raise_no_path(self, target_id: str) -> None:
        raise JunctionDesignError(
            f"Target {target_id!r} has no candidate path that meets the declared fragment synthesis "
            f"minimum of {self.minimum_fragment_oligo_length} nt."
        )


__all__ = ["FragmentPathConstraint"]
