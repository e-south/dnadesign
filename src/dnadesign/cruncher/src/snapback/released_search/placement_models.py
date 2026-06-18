"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/released_search/placement_models.py

Placement-side dataclasses for released-product target-search.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.cruncher.nickases.models import NickaseCatalogEntry
from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeEntry
from dnadesign.cruncher.snapback.released_route_policy import ReleasedActiveStrand


@dataclass(frozen=True)
class NickPlacement:
    entry: NickaseCatalogEntry
    orientation: str
    motif: str
    site_start_at_boundary_zero: int
    left_of_origin_slack_nt: int = 0

    def site_start_for_boundary(self, boundary: int) -> int:
        return self.site_start_at_boundary_zero + boundary

    def site_end_for_boundary(self, boundary: int) -> int:
        return self.site_start_for_boundary(boundary) + len(self.motif)

    def earliest_allowed_boundary(self) -> int:
        return max(0, -self.site_start_at_boundary_zero - self.left_of_origin_slack_nt)

    def allows_left_of_origin_prefix(self, boundary: int) -> bool:
        return self.site_start_for_boundary(boundary) >= -self.left_of_origin_slack_nt


@dataclass(frozen=True)
class ReleasePlacement:
    entry: ReleaseEnzymeEntry
    orientation: str
    motif: str
    site_shift_from_boundary: int
    top_cut_shift_from_boundary: int
    bottom_cut_shift_from_boundary: int
    active_strand: ReleasedActiveStrand = "bottom"

    def site_start_for_boundary(self, boundary: int) -> int:
        return boundary + self.site_shift_from_boundary

    def site_end_for_boundary(self, boundary: int) -> int:
        return self.site_start_for_boundary(boundary) + len(self.motif)

    def active_cut_for_boundary(self, boundary: int) -> int:
        if self.active_strand == "top":
            return self.top_cut_for_boundary(boundary)
        return self.bottom_cut_for_boundary(boundary)

    def top_cut_for_boundary(self, boundary: int) -> int:
        return boundary + self.top_cut_shift_from_boundary

    def bottom_cut_for_boundary(self, boundary: int) -> int:
        return boundary + self.bottom_cut_shift_from_boundary

    def starts_downstream_of_boundary(self) -> bool:
        return self.site_shift_from_boundary >= 0


@dataclass(frozen=True)
class BuiltPrecursor:
    top_strand: str
    coordinate_offset: int


@dataclass(frozen=True)
class BuiltPrecursorResult:
    precursor: BuiltPrecursor | None
    blocker_code: str | None = None


__all__ = [
    "BuiltPrecursor",
    "BuiltPrecursorResult",
    "NickPlacement",
    "ReleasePlacement",
]
