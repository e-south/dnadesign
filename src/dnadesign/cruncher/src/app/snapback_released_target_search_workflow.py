"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/snapback_released_target_search_workflow.py

Application orchestration for released-product snapback target-search.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.nickases.catalog import load_merged_nickase_catalog
from dnadesign.cruncher.release_enzymes.catalog import load_merged_release_enzyme_catalog
from dnadesign.cruncher.snapback.catalog_sources import catalog_source_label
from dnadesign.cruncher.snapback.released_models import SingleNickReleasedTargetSearchRequest
from dnadesign.cruncher.snapback.released_target_search import search_released_target_hits


def run_released_snapback_target_search(
    *,
    request: SingleNickReleasedTargetSearchRequest,
    workspace_root: Path,
):
    nick_catalog, nick_resolved_paths = load_merged_nickase_catalog(
        preset_id=request.nick_sources.preset,
        additional_preset_ids=request.nick_sources.additional_presets,
        additional_paths=request.nick_sources.additional_paths,
        workspace_root=workspace_root,
    )
    release_catalog, release_resolved_paths = load_merged_release_enzyme_catalog(
        preset_id=request.release_sources.preset,
        additional_preset_ids=request.release_sources.additional_presets,
        additional_paths=request.release_sources.additional_paths,
        workspace_root=workspace_root,
    )
    return search_released_target_hits(
        request=request,
        nick_catalog=nick_catalog,
        release_catalog=release_catalog,
        workspace_root=workspace_root,
        nick_catalog_source=catalog_source_label(
            preset_ids=request.nick_sources.resolved_preset_ids(),
            resolved_paths=nick_resolved_paths,
        ),
        release_catalog_source=catalog_source_label(
            preset_ids=request.release_sources.resolved_preset_ids(),
            resolved_paths=release_resolved_paths,
        ),
    )


__all__ = ["run_released_snapback_target_search"]
