"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/contracts/job_invariants.py

Cross-field job invariants for construct configs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable, Protocol


class PartLike(Protocol):
    name: str
    role: str


def require_realize_focal_contract(
    *,
    parts: Iterable[PartLike],
    focal_part: str | None,
    realize_mode: str,
) -> str:
    part_list = list(parts)
    part_names = {part.name for part in part_list}
    focal = str(focal_part or "").strip()
    if focal and focal not in part_names:
        raise ValueError(f"realize.focal_part '{focal_part}' is not defined in job.parts.")
    if realize_mode == "window" and focal not in part_names:
        raise ValueError(f"realize.focal_part '{focal_part}' is not defined in job.parts.")
    anchor_part_names = [part.name for part in part_list if part.role == "anchor" or part.name == "anchor"]
    if not focal and len(anchor_part_names) > 1:
        joined = ", ".join(anchor_part_names)
        raise ValueError(
            "job.parts defines multiple anchor parts "
            f"({joined}); realize.focal_part is required to choose the emitted anchor handoff span."
        )
    return focal
