"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/structure_browser_common.py

Shared helpers for Eco1 review-deliverable structure-browser manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REFERENCE_STRUCTURE_RELATIVE_PATH = "structures/ec86kit_chain_a_backbone_reference.pdb"
REFERENCE_COLOR = "#efece3"


def display_label(candidate_id: str, row: dict[str, Any]) -> str:
    if candidate_id == "wild_type":
        return "WT ColabFold baseline"
    label = str(row.get("display_label") or "")
    if label:
        return label
    return f"ProteinMPNN variant {candidate_id.removeprefix('thread_candidate_')[:12]}"


def relative_path(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def repo_relative_hint(path: Path) -> str:
    if path.parent.name == "foldcheck_review":
        return str(Path("foldcheck_review") / path.name)
    return path.name


def nullable_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def nullable_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
