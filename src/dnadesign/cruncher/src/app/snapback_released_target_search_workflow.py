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

from dnadesign.cruncher.app.snapback_released_catalogs import resolve_released_catalogs
from dnadesign.cruncher.snapback.released_models import SingleNickReleasedTargetSearchRequest
from dnadesign.cruncher.snapback.released_target_search import search_released_target_hits


def run_released_snapback_target_search(
    *,
    request: SingleNickReleasedTargetSearchRequest,
    workspace_root: Path,
):
    resolved_catalogs = resolve_released_catalogs(
        nick_sources=request.nick_sources,
        release_sources=request.release_sources,
        workspace_root=workspace_root,
    )
    return search_released_target_hits(
        request=request,
        nick_catalog=resolved_catalogs.nick_catalog,
        release_catalog=resolved_catalogs.release_catalog,
        workspace_root=workspace_root,
        nick_catalog_source=resolved_catalogs.nick_catalog_source,
        release_catalog_source=resolved_catalogs.release_catalog_source,
    )


__all__ = ["run_released_snapback_target_search"]
