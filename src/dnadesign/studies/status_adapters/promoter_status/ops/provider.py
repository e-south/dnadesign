"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/status_adapters/promoter_status/ops/provider.py

OPS status-provider entrypoints for the explicit promoter study adapter.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..adapter import STUDY_STATUS_ADAPTER


def provide_promoter_status(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    context = STUDY_STATUS_ADAPTER.load_context(
        repo_root=repo_root,
        study_root=inputs.get("study_dir"),
    )
    return STUDY_STATUS_ADAPTER.build_snapshot(context)


def provide_promoter_preflight(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    context = STUDY_STATUS_ADAPTER.load_context(
        repo_root=repo_root,
        study_root=inputs.get("study_dir"),
    )
    return STUDY_STATUS_ADAPTER.build_preflight(
        context,
        scope=inputs.get("scope"),
        command_timeout_seconds=inputs.get("command_timeout_seconds"),
    )


__all__ = [
    "provide_promoter_preflight",
    "provide_promoter_status",
]
