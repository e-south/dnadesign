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


class OutputVariantLike(Protocol):
    recommended_pooling: str | None
    anchor_part: str | None


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


def require_output_variant_anchor_handoff_contract(
    *,
    parts: Iterable[PartLike],
    focal_part: str | None,
    output_variants: Iterable[OutputVariantLike],
) -> None:
    variants = list(output_variants)
    if not variants:
        return
    focal = str(focal_part or "").strip()
    part_list = list(parts)
    part_names = {part.name for part in part_list}
    anchor_part_names = [part.name for part in part_list if part.role == "anchor" or part.name == "anchor"]
    for variant in variants:
        variant_anchor_part = str(variant.anchor_part or "").strip()
        if variant_anchor_part and variant_anchor_part not in part_names:
            raise ValueError(f"job.output_variants anchor_part '{variant_anchor_part}' is not defined in job.parts.")
        needs_anchor_handoff = variant.recommended_pooling == "anchor_mean" or bool(variant_anchor_part)
        if not needs_anchor_handoff:
            continue
        if focal or variant_anchor_part or anchor_part_names:
            continue
        raise ValueError(
            "job.output_variants requires realize.focal_part or output_variants[].anchor_part when job.parts has no "
            "part named or role 'anchor'. realized_context sequence views with anchor_mean need a declared handoff "
            "span for construct__anchor_start/end or sequence-view anchor_start/end."
        )
