"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/released_route_policy.py

Route-policy literals and validators for released-product snapback.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

_DEFAULT_DISALLOWED_NICKASE_WARNING_CODES = ("FREQUENT_CUTTER",)
_DEFAULT_ALLOWED_ACTIVE_STRANDS = ("bottom",)
_DEFAULT_ALLOWED_ROUTE_FAMILIES = ("bottom_active_from_top_nick",)

ReleasedActiveStrand = Literal["top", "bottom"]
ReleasedRouteFamily = Literal["bottom_active_from_top_nick", "top_active_from_bottom_nick"]
ReleasedFinalGeometrySource = Literal["exposed_bottom_strand", "retained_active_strand"]


@dataclass(frozen=True)
class ReleasedRouteSemantics:
    route_family: ReleasedRouteFamily
    final_geometry_source: ReleasedFinalGeometrySource
    active_strand: ReleasedActiveStrand
    retained_partner_strand: ReleasedActiveStrand
    physical_nicked_strand: ReleasedActiveStrand


@dataclass(frozen=True)
class ReleasedSearchRoutePolicy:
    final_geometry_source: ReleasedFinalGeometrySource
    allowed_active_strands: tuple[ReleasedActiveStrand, ...]
    allowed_route_families: tuple[ReleasedRouteFamily, ...]


def normalize_release_catalog_path_list(value: list[Path], *, label: str) -> list[Path]:
    normalized = [Path(path) for path in value]
    if len({str(path) for path in normalized}) != len(normalized):
        raise ValueError(f"{label} must not repeat values.")
    return normalized


def normalize_warning_code_list(value: list[str], *, label: str) -> list[str]:
    normalized = [str(item or "").strip() for item in value]
    if any(not item for item in normalized):
        raise ValueError(f"{label} must not contain blank values.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must not repeat values.")
    return normalized


def normalize_variant_id_list(value: list[str], *, label: str) -> list[str]:
    normalized = [str(item or "").strip() for item in value]
    if any(not item for item in normalized):
        raise ValueError(f"{label} must not contain blank values.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must not repeat values.")
    return normalized


def normalize_active_strand_list(value: list[str], *, label: str) -> list[ReleasedActiveStrand]:
    normalized = [str(item or "").strip() for item in value]
    if any(item not in {"top", "bottom"} for item in normalized):
        raise ValueError(f"{label} must contain only 'top' or 'bottom'.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must not repeat values.")
    return list(normalized)  # type: ignore[return-value]


def normalize_route_family_list(value: list[str], *, label: str) -> list[ReleasedRouteFamily]:
    normalized = [str(item or "").strip() for item in value]
    allowed = {"bottom_active_from_top_nick", "top_active_from_bottom_nick"}
    if any(item not in allowed for item in normalized):
        raise ValueError(f"{label} must contain only bottom_active_from_top_nick or top_active_from_bottom_nick.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must not repeat values.")
    return list(normalized)  # type: ignore[return-value]


def released_route_semantics(route_family: ReleasedRouteFamily) -> ReleasedRouteSemantics:
    if route_family == "bottom_active_from_top_nick":
        return ReleasedRouteSemantics(
            route_family=route_family,
            final_geometry_source="exposed_bottom_strand",
            active_strand="bottom",
            retained_partner_strand="top",
            physical_nicked_strand="top",
        )
    return ReleasedRouteSemantics(
        route_family=route_family,
        final_geometry_source="retained_active_strand",
        active_strand="top",
        retained_partner_strand="bottom",
        physical_nicked_strand="bottom",
    )


def route_family_active_strand(route_family: ReleasedRouteFamily) -> ReleasedActiveStrand:
    return released_route_semantics(route_family).active_strand


def route_family_retained_partner_strand(route_family: ReleasedRouteFamily) -> ReleasedActiveStrand:
    return released_route_semantics(route_family).retained_partner_strand


def route_family_physical_nicked_strand(route_family: ReleasedRouteFamily) -> ReleasedActiveStrand:
    return released_route_semantics(route_family).physical_nicked_strand


def route_family_final_geometry_source(route_family: ReleasedRouteFamily) -> ReleasedFinalGeometrySource:
    return released_route_semantics(route_family).final_geometry_source


def released_search_route_policy(*, allow_top_active_routes: bool) -> ReleasedSearchRoutePolicy:
    if allow_top_active_routes:
        return ReleasedSearchRoutePolicy(
            final_geometry_source="retained_active_strand",
            allowed_active_strands=("top", "bottom"),
            allowed_route_families=("bottom_active_from_top_nick", "top_active_from_bottom_nick"),
        )
    return ReleasedSearchRoutePolicy(
        final_geometry_source="exposed_bottom_strand",
        allowed_active_strands=_DEFAULT_ALLOWED_ACTIVE_STRANDS,
        allowed_route_families=_DEFAULT_ALLOWED_ROUTE_FAMILIES,
    )


def infer_released_search_final_geometry_source(
    *,
    allowed_active_strands: Iterable[ReleasedActiveStrand | str],
    allowed_route_families: Iterable[ReleasedRouteFamily | str],
) -> ReleasedFinalGeometrySource:
    normalized_active_strands = normalize_active_strand_list(
        [str(item) for item in allowed_active_strands],
        label="allowed_active_strands",
    )
    normalized_route_families = normalize_route_family_list(
        [str(item) for item in allowed_route_families],
        label="allowed_route_families",
    )
    route_active_strands = {route_family_active_strand(route) for route in normalized_route_families}
    if not route_active_strands.issubset(set(normalized_active_strands)):
        raise ValueError("allowed_route_families must be compatible with allowed_active_strands.")
    route_geometry_sources = {route_family_final_geometry_source(route) for route in normalized_route_families}
    if route_geometry_sources == {"exposed_bottom_strand"}:
        return "exposed_bottom_strand"
    return "retained_active_strand"


__all__ = [
    "ReleasedActiveStrand",
    "ReleasedFinalGeometrySource",
    "ReleasedRouteFamily",
    "ReleasedRouteSemantics",
    "ReleasedSearchRoutePolicy",
    "_DEFAULT_ALLOWED_ACTIVE_STRANDS",
    "_DEFAULT_ALLOWED_ROUTE_FAMILIES",
    "_DEFAULT_DISALLOWED_NICKASE_WARNING_CODES",
    "infer_released_search_final_geometry_source",
    "normalize_active_strand_list",
    "normalize_release_catalog_path_list",
    "normalize_route_family_list",
    "normalize_warning_code_list",
    "route_family_active_strand",
    "route_family_final_geometry_source",
    "route_family_physical_nicked_strand",
    "route_family_retained_partner_strand",
    "released_route_semantics",
    "released_search_route_policy",
]
