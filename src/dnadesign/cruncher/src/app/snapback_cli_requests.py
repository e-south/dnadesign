"""
Typed CLI-to-app request builders for Snapback workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.cruncher.snapback.models import CatalogSources
from dnadesign.cruncher.snapback.released_models import (
    ReleaseCatalogSources,
    ReleasedFinalTargetGeometry,
    ReleasedSolveOutputConfig,
    ReleasedTargetSearchConfig,
    SingleNickReleasedTargetSearchRequest,
)
from dnadesign.cruncher.snapback.released_route_policy import released_search_route_policy
from dnadesign.cruncher.snapback.target_models import SnapbackTargetGeometry


@dataclass(frozen=True)
class SnapbackTargetSearchInvocation:
    catalog: CatalogSources
    workspace_root: Path
    target: SnapbackTargetGeometry
    normalize_to_top_strand_nick: bool
    max_results: int


@dataclass(frozen=True)
class ReleasedTargetSearchInvocation:
    request: SingleNickReleasedTargetSearchRequest
    workspace_root: Path


@dataclass(frozen=True)
class ReleasedSolveInvocation:
    request: SingleNickReleasedTargetSearchRequest
    output: ReleasedSolveOutputConfig
    workspace_root: Path


def resolve_workspace_root(workspace_root: Path) -> Path:
    return workspace_root.expanduser().resolve()


def build_snapback_target_search_invocation(
    *,
    preset: str | None,
    additional_preset: list[str],
    additional_path: list[Path],
    workspace_root: Path,
    nick_boundary: int,
    paired_bp: int,
    cap_nt: int,
    max_results: int,
    normalize_to_top_strand_nick: bool,
) -> SnapbackTargetSearchInvocation:
    effective_preset = preset
    effective_additional_presets = list(additional_preset)
    if effective_preset is None and not effective_additional_presets and not additional_path:
        effective_preset = "neb_nicking_v1"
        effective_additional_presets = ["thermo_nicking_v1"]
    return SnapbackTargetSearchInvocation(
        catalog=CatalogSources(
            preset=effective_preset,
            additional_presets=effective_additional_presets,
            additional_paths=additional_path,
        ),
        workspace_root=resolve_workspace_root(workspace_root),
        target=SnapbackTargetGeometry(
            nick_boundary_from_left=nick_boundary,
            paired_bp=paired_bp,
            cap_nt=cap_nt,
            require_site_sequence_preserved=True,
        ),
        normalize_to_top_strand_nick=normalize_to_top_strand_nick,
        max_results=max_results,
    )


def _released_disallowed_warning_codes(allow_frequent_cutter_nickases: bool) -> list[str]:
    if allow_frequent_cutter_nickases:
        return []
    return ["FREQUENT_CUTTER"]


def _validate_released_explicit_sources(
    *,
    workflow_label: str,
    nick_preset: str | None,
    nick_additional_preset: list[str],
    nick_additional_path: list[Path],
    release_preset: str | None,
    release_additional_preset: list[str],
    release_additional_path: list[Path],
) -> None:
    if nick_preset is None and not nick_additional_preset and not nick_additional_path:
        raise ValueError(
            f"{workflow_label} requires at least one explicit nickase source "
            "(--nick-preset, --nick-additional-preset, or --nick-additional-path)."
        )
    if release_preset is None and not release_additional_preset and not release_additional_path:
        raise ValueError(
            f"{workflow_label} requires at least one explicit release-enzyme source "
            "(--release-preset, --release-additional-preset, or --release-additional-path)."
        )


def build_released_target_search_invocation(
    *,
    nick_preset: str | None,
    nick_additional_preset: list[str],
    nick_additional_path: list[Path],
    release_preset: str | None,
    release_additional_preset: list[str],
    release_additional_path: list[Path],
    workspace_root: Path,
    nick_boundary: int,
    paired_bp: int,
    cap_nt: int,
    max_results: int,
    near_boundary_search_limit: int,
    release_variant_id: list[str],
    allow_demo_hits: bool,
    allow_frequent_cutter_nickases: bool,
    allow_top_active_routes: bool,
    allow_precut_footprint_outside_active_product: bool,
) -> ReleasedTargetSearchInvocation:
    _validate_released_explicit_sources(
        workflow_label="released-target-search",
        nick_preset=nick_preset,
        nick_additional_preset=nick_additional_preset,
        nick_additional_path=nick_additional_path,
        release_preset=release_preset,
        release_additional_preset=release_additional_preset,
        release_additional_path=release_additional_path,
    )
    route_policy = released_search_route_policy(allow_top_active_routes=allow_top_active_routes)
    return ReleasedTargetSearchInvocation(
        request=SingleNickReleasedTargetSearchRequest(
            target=ReleasedFinalTargetGeometry(
                nick_boundary_from_left=nick_boundary,
                paired_bp=paired_bp,
                cap_nt=cap_nt,
            ),
            nick_sources=CatalogSources(
                preset=nick_preset,
                additional_presets=nick_additional_preset,
                additional_paths=nick_additional_path,
            ),
            release_sources=ReleaseCatalogSources(
                preset=release_preset,
                additional_presets=release_additional_preset,
                additional_paths=release_additional_path,
            ),
            search=ReleasedTargetSearchConfig(
                route_policy_final_geometry_source=route_policy.final_geometry_source,
                max_results=max_results,
                near_boundary_search_limit=near_boundary_search_limit,
                allow_demo_hits=allow_demo_hits,
                allowed_release_variant_ids=release_variant_id,
                allow_precut_footprint_outside_active_product=allow_precut_footprint_outside_active_product,
                allowed_active_strands=list(route_policy.allowed_active_strands),
                allowed_route_families=list(route_policy.allowed_route_families),
                disallowed_nickase_warning_codes=_released_disallowed_warning_codes(allow_frequent_cutter_nickases),
            ),
        ),
        workspace_root=resolve_workspace_root(workspace_root),
    )


def build_released_solve_invocation(
    *,
    nick_preset: str | None,
    nick_additional_preset: list[str],
    nick_additional_path: list[Path],
    release_preset: str | None,
    release_additional_preset: list[str],
    release_additional_path: list[Path],
    workspace_root: Path,
    nick_boundary: int,
    paired_bp: int,
    cap_nt: int,
    max_results: int,
    near_boundary_search_limit: int,
    materialize_top_k: int,
    release_variant_id: list[str],
    run_dir: Path,
    render_format: str,
    emit_renders: bool,
    allow_demo_hits: bool,
    allow_frequent_cutter_nickases: bool,
    allow_top_active_routes: bool,
    allow_precut_footprint_outside_active_product: bool,
) -> ReleasedSolveInvocation:
    _validate_released_explicit_sources(
        workflow_label="released-solve",
        nick_preset=nick_preset,
        nick_additional_preset=nick_additional_preset,
        nick_additional_path=nick_additional_path,
        release_preset=release_preset,
        release_additional_preset=release_additional_preset,
        release_additional_path=release_additional_path,
    )
    route_policy = released_search_route_policy(allow_top_active_routes=allow_top_active_routes)
    return ReleasedSolveInvocation(
        request=SingleNickReleasedTargetSearchRequest(
            target=ReleasedFinalTargetGeometry(
                nick_boundary_from_left=nick_boundary,
                paired_bp=paired_bp,
                cap_nt=cap_nt,
            ),
            nick_sources=CatalogSources(
                preset=nick_preset,
                additional_presets=nick_additional_preset,
                additional_paths=nick_additional_path,
            ),
            release_sources=ReleaseCatalogSources(
                preset=release_preset,
                additional_presets=release_additional_preset,
                additional_paths=release_additional_path,
            ),
            search=ReleasedTargetSearchConfig(
                route_policy_final_geometry_source=route_policy.final_geometry_source,
                max_results=max(max_results, materialize_top_k),
                near_boundary_search_limit=near_boundary_search_limit,
                allow_demo_hits=allow_demo_hits,
                allowed_release_variant_ids=release_variant_id,
                allow_precut_footprint_outside_active_product=allow_precut_footprint_outside_active_product,
                allowed_active_strands=list(route_policy.allowed_active_strands),
                allowed_route_families=list(route_policy.allowed_route_families),
                disallowed_nickase_warning_codes=_released_disallowed_warning_codes(allow_frequent_cutter_nickases),
            ),
        ),
        output=ReleasedSolveOutputConfig(
            run_dir=run_dir,
            materialize_top_k=materialize_top_k,
            render_format=render_format,
            emit_renders=emit_renders,
        ),
        workspace_root=resolve_workspace_root(workspace_root),
    )


__all__ = [
    "ReleasedSolveInvocation",
    "ReleasedTargetSearchInvocation",
    "SnapbackTargetSearchInvocation",
    "build_released_solve_invocation",
    "build_released_target_search_invocation",
    "build_snapback_target_search_invocation",
    "resolve_workspace_root",
]
