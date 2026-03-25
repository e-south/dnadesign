"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/stress_promoter_ethanol_cipro/ops_provider.py

OPS status-provider entrypoints for the explicit stress_promoter_ethanol_cipro
study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .family import STRESS_PROMOTER_ETHANOL_CIPRO_STUDY_ADAPTER


def provide_stress_promoter_ethanol_cipro_status(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    context = STRESS_PROMOTER_ETHANOL_CIPRO_STUDY_ADAPTER.load_context(
        repo_root=repo_root,
        study_root=inputs.get("study_dir"),
    )
    return STRESS_PROMOTER_ETHANOL_CIPRO_STUDY_ADAPTER.build_snapshot(context)


def provide_stress_promoter_ethanol_cipro_preflight(
    *,
    repo_root: Path | None,
    inputs: Mapping[str, object],
) -> tuple[str, str, dict[str, object]]:
    context = STRESS_PROMOTER_ETHANOL_CIPRO_STUDY_ADAPTER.load_context(
        repo_root=repo_root,
        study_root=inputs.get("study_dir"),
    )
    return STRESS_PROMOTER_ETHANOL_CIPRO_STUDY_ADAPTER.build_preflight(
        context,
        scope=inputs.get("scope"),
    )


__all__ = [
    "provide_stress_promoter_ethanol_cipro_preflight",
    "provide_stress_promoter_ethanol_cipro_status",
]
