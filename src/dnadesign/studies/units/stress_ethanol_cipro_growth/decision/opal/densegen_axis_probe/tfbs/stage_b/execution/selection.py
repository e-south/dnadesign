"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/execution/selection.py

Selection artifact contracts for Stage B execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ....core.selection_artifacts import probe_selection_path, read_probe_selection
from ..semantics import (
    TFBS_STAGE_B_EXACT_BUDGET_TIE_HANDLING,
    validate_stage_b_tie_handling,
)


def selected_ids_from_round(
    *,
    workdir: Path,
    round_index: int,
    selection_k: int | None = None,
    tie_handling: str | None = None,
) -> tuple[str, ...]:
    path = probe_selection_path(workdir, round_index)
    if not path.exists():
        raise FileNotFoundError(f"Stage B selection artifact missing: {path}")
    frame = read_probe_selection(workdir, round_index)
    ids = tuple(str(value).strip() for value in frame["id"].tolist())
    if not ids or any(not value for value in ids):
        raise ValueError(f"Stage B selection artifact has blank or empty ids: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Stage B selection artifact contains duplicate ids: {path}")
    if selection_k is not None and tie_handling is not None:
        assert_selected_count(
            selected_count=len(ids),
            path=path,
            selection_k=selection_k,
            tie_handling=tie_handling,
        )
    return ids


def selection_exists(*, workdir: Path, round_index: int) -> bool:
    return probe_selection_path(workdir, round_index).exists()


def assert_selection_budget(*, workdir: Path, round_index: int, selection_k: int, tie_handling: str) -> None:
    path = probe_selection_path(workdir, round_index)
    if not path.exists():
        raise FileNotFoundError(f"Stage B selection artifact missing: {path}")
    frame = read_probe_selection(workdir, round_index)
    assert_selected_count(
        selected_count=len(frame),
        path=path,
        selection_k=selection_k,
        tie_handling=tie_handling,
    )


def assert_selected_count(*, selected_count: int, path: Path, selection_k: int, tie_handling: str) -> None:
    tie = validate_stage_b_tie_handling(tie_handling)
    if int(selection_k) <= 0:
        raise ValueError("Stage B selection_k must be positive")
    if tie != TFBS_STAGE_B_EXACT_BUDGET_TIE_HANDLING:
        return
    if int(selected_count) != int(selection_k):
        raise RuntimeError(
            "Stage B exact-budget selection contract failed: "
            f"expected {int(selection_k)} selected row(s), observed {int(selected_count)} in {path}"
        )


def campaign_selection_k(campaign: Mapping[str, Any]) -> int:
    value = int(campaign.get("selection_k", 0))
    if value <= 0:
        raise ValueError(f"Stage B campaign missing positive selection_k: {campaign.get('campaign_key')}")
    return value


def campaign_tie_handling(campaign: Mapping[str, Any]) -> str:
    value = campaign.get("selection_tie_handling", TFBS_STAGE_B_EXACT_BUDGET_TIE_HANDLING)
    return validate_stage_b_tie_handling(str(value))
