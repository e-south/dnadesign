"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/retron_hairpin_design/status/ops/provider.py

OPS status-provider entrypoints for the Retron hairpin design status service.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def provide_retron_hairpin_design_status(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    from ..service import STUDY_STATUS_SERVICE

    context = STUDY_STATUS_SERVICE.load_context(
        repo_root=repo_root,
        study_root=inputs.get("study_dir"),
    )
    return STUDY_STATUS_SERVICE.build_snapshot(context)


def provide_retron_hairpin_design_preflight(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    from ..service import STUDY_STATUS_SERVICE

    context = STUDY_STATUS_SERVICE.load_context(
        repo_root=repo_root,
        study_root=inputs.get("study_dir"),
    )
    return STUDY_STATUS_SERVICE.build_preflight(
        context,
        scope=inputs.get("scope"),
    )


__all__ = ["provide_retron_hairpin_design_preflight", "provide_retron_hairpin_design_status"]
