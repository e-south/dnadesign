"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/realization/placement_models.py

Placement value objects for construct realization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..contracts.config import PartConfig


class TemplateLike(Protocol):
    id: str
    sequence: str
    circular: bool


@dataclass(frozen=True)
class PlacementSite:
    start: int
    end: int
    locator_kind: str
    locator_upstream_sequence: str | None
    locator_downstream_sequence: str | None


@dataclass(frozen=True)
class PlacementPlan:
    part: PartConfig
    site: PlacementSite


@dataclass(frozen=True)
class PlannedPlacement:
    part_name: str
    part_role: str
    sequence_source: str
    sequence_field: str | None
    placement_kind: str
    template_start: int
    template_end: int
    template_span_bp: int
    orientation: str
    locator_kind: str
    locator_upstream_sequence: str | None
    locator_downstream_sequence: str | None
    guard_mode: str
    guard_require_unique_forward_matches: bool
    guard_replaced_span_bp: int | None
    template_sequence: str
    guard_replaced_sequence: str | None
    guard_upstream_sequence: str | None
    observed_guard_upstream_sequence: str | None
    guard_downstream_sequence: str | None
    observed_guard_downstream_sequence: str | None
