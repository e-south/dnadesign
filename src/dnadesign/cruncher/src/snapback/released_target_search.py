"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_target_search.py

Target-first paired nickase plus release-enzyme search for released-product
snapback designs.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.nickases.models import NickaseCatalog
from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeCatalog
from dnadesign.cruncher.snapback.released_projection import evaluate_released_precursor
from dnadesign.cruncher.snapback.released_search.evaluator_adapter import (
    hit_from_evaluation as _hit_from_evaluation_impl,
)
from dnadesign.cruncher.snapback.released_search.evaluator_adapter import (
    search_pair as _search_pair_impl,
)
from dnadesign.cruncher.snapback.released_search.nick_placements import (
    nick_placements as _nick_placements_impl,
)
from dnadesign.cruncher.snapback.released_search.nick_placements import (
    nickase_entry_has_disallowed_warning_code as _nickase_entry_has_disallowed_warning_code_impl,
)
from dnadesign.cruncher.snapback.released_search.nick_placements import (
    nickase_entry_is_demo_only as _nickase_entry_is_demo_only_impl,
)
from dnadesign.cruncher.snapback.released_search.precursor_builder import (
    build_precursor_sequence as _build_precursor_sequence_impl,
)
from dnadesign.cruncher.snapback.released_search.ranking import ReleasedRankingPolicy
from dnadesign.cruncher.snapback.released_search.ranking import rank_hits as _rank_hits_impl
from dnadesign.cruncher.snapback.released_search.release_placements import (
    release_entry_is_demo_only as _release_entry_is_demo_only_impl,
)
from dnadesign.cruncher.snapback.released_search.release_placements import (
    release_placements as _release_placements_impl,
)
from dnadesign.cruncher.snapback.released_search.reporting import blocker as _blocker_impl
from dnadesign.cruncher.snapback.released_search.runner import search_released_target_hits as _search_runner
from dnadesign.cruncher.snapback.released_search_models import (
    ReleasedTargetSearchHit,
    ReleasedTargetSearchReport,
    SingleNickReleasedTargetSearchRequest,
)


def _blocker(counts: dict[str, int], code: str) -> None:
    _blocker_impl(counts, code)


def _nickase_entry_is_demo_only(entry) -> bool:
    return _nickase_entry_is_demo_only_impl(entry)


def _release_entry_is_demo_only(entry) -> bool:
    return _release_entry_is_demo_only_impl(entry)


def _nickase_entry_has_disallowed_warning_code(entry, *, warning_codes: list[str]) -> bool:
    return _nickase_entry_has_disallowed_warning_code_impl(entry, warning_codes=warning_codes)


def _nick_placements(*args, **kwargs):
    return _nick_placements_impl(*args, **kwargs)


def _release_placements(*args, **kwargs):
    return _release_placements_impl(*args, **kwargs)


def _build_precursor_sequence(*args, **kwargs):
    return _build_precursor_sequence_impl(*args, **kwargs)


def _hit_from_evaluation(*args, **kwargs):
    return _hit_from_evaluation_impl(*args, **kwargs)


def _rank_hits(
    hits: list[ReleasedTargetSearchHit],
    *,
    target,
    exact: bool,
) -> list[ReleasedTargetSearchHit]:
    return _rank_hits_impl(hits, target=target, exact=exact)


def _search_pair(
    *,
    request,
    route_family="bottom_active_from_top_nick",
    nick_placement,
    release_placement,
    blocker_counts,
):
    return _search_pair_impl(
        request=request,
        route_family=route_family,
        nick_placement=nick_placement,
        release_placement=release_placement,
        blocker_counts=blocker_counts,
        build_precursor_sequence_fn=_build_precursor_sequence,
        evaluate_released_precursor_fn=evaluate_released_precursor,
        hit_from_evaluation_fn=_hit_from_evaluation,
        blocker_fn=_blocker,
    )


def search_released_target_hits(
    *,
    request: SingleNickReleasedTargetSearchRequest,
    nick_catalog: NickaseCatalog,
    release_catalog: ReleaseEnzymeCatalog,
    workspace_root: Path,
    nick_catalog_source: str,
    release_catalog_source: str,
) -> ReleasedTargetSearchReport:
    return _search_runner(
        request=request,
        nick_catalog=nick_catalog,
        release_catalog=release_catalog,
        workspace_root=workspace_root,
        nick_catalog_source=nick_catalog_source,
        release_catalog_source=release_catalog_source,
        nick_placements_fn=_nick_placements,
        release_placements_fn=_release_placements,
        search_pair_fn=_search_pair,
        rank_hits_fn=_rank_hits,
        nickase_entry_is_demo_only_fn=_nickase_entry_is_demo_only,
        release_entry_is_demo_only_fn=_release_entry_is_demo_only,
        nickase_entry_has_disallowed_warning_code_fn=_nickase_entry_has_disallowed_warning_code,
        blocker_fn=_blocker,
    )


__all__ = [
    "ReleasedRankingPolicy",
    "search_released_target_hits",
]
