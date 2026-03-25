"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/promoter/ops_provider.py

OPS status-provider entrypoints for the explicit promoter
study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .family import PROMOTER_STUDY_ADAPTER


def provide_promoter_status(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    context = PROMOTER_STUDY_ADAPTER.load_context(
        repo_root=repo_root,
        study_root=inputs.get("study_dir"),
    )
    return PROMOTER_STUDY_ADAPTER.build_snapshot(context)


def provide_promoter_preflight(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    context = PROMOTER_STUDY_ADAPTER.load_context(
        repo_root=repo_root,
        study_root=inputs.get("study_dir"),
    )
    return PROMOTER_STUDY_ADAPTER.build_preflight(
        context,
        scope=inputs.get("scope"),
    )


__all__ = [
    "provide_promoter_preflight",
    "provide_promoter_status",
]
