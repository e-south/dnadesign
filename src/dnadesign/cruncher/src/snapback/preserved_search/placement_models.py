"""
Placement models for preserved-site target search.
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.cruncher.nickases.models import NickaseCatalogEntry


@dataclass(frozen=True)
class Placement:
    entry: NickaseCatalogEntry
    orientation: str
    motif: str
    site_start_at_target_boundary: int
    boundary_offset: int
    exact_input_length_nt: int | None
    earliest_feasible_boundary: int | None
    earliest_input_length_nt: int | None
    exact_boundary_blockers: tuple[str, ...]

    @property
    def exact_boundary_hit_possible(self) -> bool:
        return self.exact_input_length_nt is not None

    @property
    def any_boundary_hit_possible(self) -> bool:
        return self.earliest_feasible_boundary is not None

    def site_start_for_boundary(self, boundary: int) -> int:
        return boundary - self.boundary_offset
