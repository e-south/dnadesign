"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_route_policy.py

Route-policy literals and validators for released-product snapback.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

_DEFAULT_DISALLOWED_NICKASE_WARNING_CODES = ("FREQUENT_CUTTER",)
_DEFAULT_ALLOWED_ACTIVE_STRANDS = ("bottom",)
_DEFAULT_ALLOWED_ROUTE_FAMILIES = ("bottom_active_from_top_nick",)

ReleasedActiveStrand = Literal["top", "bottom"]
ReleasedRouteFamily = Literal["bottom_active_from_top_nick", "top_active_from_bottom_nick"]
ReleasedFinalGeometrySource = Literal["exposed_bottom_strand", "retained_active_strand"]


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


def route_family_active_strand(route_family: ReleasedRouteFamily) -> ReleasedActiveStrand:
    return "bottom" if route_family == "bottom_active_from_top_nick" else "top"


def route_family_physical_nicked_strand(route_family: ReleasedRouteFamily) -> ReleasedActiveStrand:
    return "top" if route_family == "bottom_active_from_top_nick" else "bottom"


def route_family_final_geometry_source(route_family: ReleasedRouteFamily) -> ReleasedFinalGeometrySource:
    return "exposed_bottom_strand" if route_family == "bottom_active_from_top_nick" else "retained_active_strand"


__all__ = [
    "ReleasedActiveStrand",
    "ReleasedFinalGeometrySource",
    "ReleasedRouteFamily",
    "_DEFAULT_ALLOWED_ACTIVE_STRANDS",
    "_DEFAULT_ALLOWED_ROUTE_FAMILIES",
    "_DEFAULT_DISALLOWED_NICKASE_WARNING_CODES",
    "normalize_active_strand_list",
    "normalize_release_catalog_path_list",
    "normalize_route_family_list",
    "normalize_warning_code_list",
    "route_family_active_strand",
    "route_family_final_geometry_source",
    "route_family_physical_nicked_strand",
]
