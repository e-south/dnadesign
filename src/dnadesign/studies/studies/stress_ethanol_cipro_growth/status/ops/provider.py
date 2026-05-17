"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/stress_ethanol_cipro_growth/status/ops/provider.py

OPS status-provider entrypoints for the stress_ethanol_cipro_growth status service.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..service import STUDY_STATUS_SERVICE


def provide_stress_ethanol_cipro_growth_status(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    context = STUDY_STATUS_SERVICE.load_context(
        repo_root=repo_root,
        study_root=inputs.get("study_dir"),
    )
    return STUDY_STATUS_SERVICE.build_snapshot(context)


def provide_stress_ethanol_cipro_growth_preflight(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    context = STUDY_STATUS_SERVICE.load_context(
        repo_root=repo_root,
        study_root=inputs.get("study_dir"),
    )
    return STUDY_STATUS_SERVICE.build_preflight(
        context,
        scope=inputs.get("scope"),
        command_timeout_seconds=inputs.get("command_timeout_seconds"),
    )


__all__ = [
    "provide_stress_ethanol_cipro_growth_preflight",
    "provide_stress_ethanol_cipro_growth_status",
]
