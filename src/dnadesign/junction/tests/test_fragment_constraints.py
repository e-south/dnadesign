"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/test_fragment_constraints.py

Direct contracts for fragment-order candidate-path feasibility.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.junction.design.fragment_constraints import FragmentPathConstraint
from dnadesign.junction.design.loci import ToeholdCandidate, ToeholdLocus
from dnadesign.junction.errors import JunctionDesignError


def _locus(target_id: str, locus_index: int, starts: tuple[int, ...]) -> ToeholdLocus:
    return ToeholdLocus(
        target_id=target_id,
        assembly_group_id="assembly-a",
        index=locus_index,
        candidates=tuple(
            ToeholdCandidate(
                target_id=target_id,
                assembly_group_id="assembly-a",
                locus_index=locus_index,
                candidate_offset=offset,
                start=start,
                sequence="ACGTACGT",
            )
            for offset, start in enumerate(starts)
        ),
    )


def test_first_feasible_path_fails_when_no_complete_path_meets_floor() -> None:
    constraint = FragmentPathConstraint(
        target_lengths=(("target-a", 60),),
        barcode_length=16,
        toehold_length=8,
        minimum_fragment_oligo_length=24,
    )

    with pytest.raises(JunctionDesignError, match="no candidate path.*minimum of 24 nt"):
        constraint.first_feasible_path((_locus("target-a", 0, (22, 23)),))


def test_first_feasible_path_is_lexical_and_independent_per_target() -> None:
    constraint = FragmentPathConstraint(
        target_lengths=(("target-a", 60), ("target-b", 62)),
        barcode_length=16,
        toehold_length=8,
        minimum_fragment_oligo_length=23,
    )
    loci = (
        _locus("target-b", 0, (22, 23)),
        _locus("target-a", 0, (22, 23)),
    )

    path = constraint.first_feasible_path(loci)

    assert tuple(candidate.target_id for candidate in path) == ("target-a", "target-b")
    assert tuple(candidate.candidate_offset for candidate in path) == (1, 1)
    assert constraint.allows(path)
