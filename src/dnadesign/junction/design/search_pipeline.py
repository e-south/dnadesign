"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/design/search_pipeline.py

Internal-only composition for one versioned junction string search.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

# This module is not a plugin API. The public planner always selects the single
# built-in composition; the injectable compiler seam exists only for internal
# verification and future versioned implementation work.

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Protocol as _Protocol

from dnadesign.junction.design import barcodes as _barcodes
from dnadesign.junction.design import matching as _matching
from dnadesign.junction.design import toeholds as _toeholds

if _TYPE_CHECKING:
    from dnadesign.junction.design.barcodes import BarcodeSelection
    from dnadesign.junction.design.fragment_constraints import FragmentPathConstraint
    from dnadesign.junction.design.loci import ToeholdCandidate, ToeholdLocus
    from dnadesign.junction.design.matching import MatchingSelection
    from dnadesign.junction.design.toeholds import ToeholdSelection

__all__: tuple[str, ...] = ()


class _ToeholdSelector(_Protocol):
    def __call__(
        self,
        loci: tuple[ToeholdLocus, ...],
        *,
        iterations: int,
        seed: int,
        path_constraint: FragmentPathConstraint | None = None,
    ) -> ToeholdSelection: ...


class _BarcodeCandidateGenerator(_Protocol):
    def __call__(
        self,
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
    ) -> tuple[str, ...]: ...


class _BarcodeSelector(_Protocol):
    def __call__(
        self,
        candidates: tuple[str, ...],
        *,
        count: int,
        iterations: int,
        seed: int,
        forbidden_toehold_k: int,
        forbidden_barcode_k: int,
    ) -> BarcodeSelection: ...


class _BarcodeMatcher(_Protocol):
    def __call__(
        self,
        candidates: tuple[ToeholdCandidate, ...],
        barcodes: tuple[str, ...],
        *,
        iterations: int,
        seed: int,
    ) -> MatchingSelection: ...


@_dataclass(frozen=True, slots=True)
class _SearchPipeline:
    """One immutable, identified composition of the four string-search stages."""

    algorithm_id: str
    select_toeholds: _ToeholdSelector
    generate_barcode_candidates: _BarcodeCandidateGenerator
    select_barcodes: _BarcodeSelector
    match_barcodes: _BarcodeMatcher

    def __post_init__(self) -> None:
        if (
            not self.algorithm_id
            or self.algorithm_id != self.algorithm_id.strip()
            or any(character.isspace() for character in self.algorithm_id)
        ):
            raise ValueError("Search pipeline algorithm_id must be nonempty and contain no whitespace.")
        for field_name in (
            "select_toeholds",
            "generate_barcode_candidates",
            "select_barcodes",
            "match_barcodes",
        ):
            if not callable(getattr(self, field_name)):
                raise TypeError(f"Search pipeline stage '{field_name}' must be callable.")


_STRING_SEARCH_V1 = _SearchPipeline(
    algorithm_id="dnadesign.junction.string.v1",
    select_toeholds=_toeholds.select_toeholds,
    generate_barcode_candidates=_barcodes.generate_barcode_candidates,
    select_barcodes=_barcodes.select_barcodes,
    match_barcodes=_matching.match_barcodes,
)
