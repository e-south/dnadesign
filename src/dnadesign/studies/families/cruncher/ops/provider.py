"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/families/cruncher/ops/provider.py

OPS status-provider entrypoints for the Cruncher study adapter.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..adapter import STUDY_FAMILY_ADAPTER


def provide_cruncher_status(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    context = STUDY_FAMILY_ADAPTER.load_context(
        repo_root=repo_root,
        study_root=inputs.get("study_dir"),
    )
    return STUDY_FAMILY_ADAPTER.build_snapshot(context)


def provide_cruncher_preflight(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    context = STUDY_FAMILY_ADAPTER.load_context(
        repo_root=repo_root,
        study_root=inputs.get("study_dir"),
    )
    return STUDY_FAMILY_ADAPTER.build_preflight(
        context,
        scope=inputs.get("scope"),
    )


__all__ = ["provide_cruncher_preflight", "provide_cruncher_status"]
