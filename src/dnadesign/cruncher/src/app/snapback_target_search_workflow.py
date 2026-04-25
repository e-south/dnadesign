"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/snapback_target_search_workflow.py

Application orchestration for target-first snapback catalog search.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.nickases.catalog import load_merged_nickase_catalog
from dnadesign.cruncher.snapback.models import CatalogSources
from dnadesign.cruncher.snapback.target_models import SnapbackTargetGeometry
from dnadesign.cruncher.snapback.target_search import search_snapback_target_hits


def run_snapback_target_search(
    *,
    catalog: CatalogSources,
    workspace_root: Path,
    target: SnapbackTargetGeometry,
    normalize_to_top_strand_nick: bool = True,
    max_results: int = 8,
):
    resolved_catalog, resolved_paths = load_merged_nickase_catalog(
        preset_id=catalog.preset,
        additional_preset_ids=catalog.additional_presets,
        additional_paths=catalog.additional_paths,
        workspace_root=workspace_root,
    )
    return search_snapback_target_hits(
        catalog=resolved_catalog,
        target=target,
        workspace_root=workspace_root,
        catalog_preset=catalog.preset,
        catalog_presets=catalog.resolved_preset_ids(),
        catalog_additional_paths=resolved_paths,
        normalize_to_top_strand_nick=normalize_to_top_strand_nick,
        max_results=max_results,
    )
