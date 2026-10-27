"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/yiu/visual_system.py

Named visual-system policy for payload-centric YIU bundle views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from dnadesign.cruncher.yiu.visual_directions import (
    evidence_ribbon_style_overrides,
    operator_strip_style_overrides,
)

YIU_VISUAL_SYSTEM_NAME = "bench_strip"


@dataclass(frozen=True)
class YiuViewStyleProfile:
    view_id: str
    direction_name: str
    system_name: str
    design_note: str
    style_overrides: dict[str, object]


def _build_style_profile(
    *,
    view_id: str,
    direction_name: str,
    design_note: str,
    style_overrides: dict[str, object],
) -> YiuViewStyleProfile:
    return YiuViewStyleProfile(
        view_id=view_id,
        direction_name=direction_name,
        system_name=YIU_VISUAL_SYSTEM_NAME,
        design_note=design_note,
        style_overrides=style_overrides,
    )


_STYLE_PROFILES: dict[str, YiuViewStyleProfile] = {
    "payload": _build_style_profile(
        view_id="payload",
        direction_name="evidence_ribbon",
        design_note="Dense operator-first evidence row for payload truth, mismatches, and PWM overlays.",
        style_overrides=evidence_ribbon_style_overrides(),
    ),
    "split_payload": _build_style_profile(
        view_id="split_payload",
        direction_name="operator_strip",
        design_note="Lean assembly strip that keeps split-fragment geometry readable without payload-row ornament.",
        style_overrides=operator_strip_style_overrides(),
    ),
    "assembled_payload": _build_style_profile(
        view_id="assembled_payload",
        direction_name="operator_strip",
        design_note="Lean reassembly strip that centers the restored payload order and junction context.",
        style_overrides=operator_strip_style_overrides(padding_y=28.0),
    ),
}


def get_yiu_style_profile(view_id: str) -> YiuViewStyleProfile:
    try:
        profile = _STYLE_PROFILES[view_id]
    except KeyError as exc:
        supported = ", ".join(sorted(_STYLE_PROFILES))
        raise ValueError(f"unsupported YIU view id {view_id!r}; expected one of: {supported}") from exc
    return YiuViewStyleProfile(
        view_id=profile.view_id,
        direction_name=profile.direction_name,
        system_name=profile.system_name,
        design_note=profile.design_note,
        style_overrides=deepcopy(profile.style_overrides),
    )


def build_yiu_style_overrides(view_id: str) -> dict[str, object]:
    return get_yiu_style_profile(view_id).style_overrides


__all__ = [
    "build_yiu_style_overrides",
    "get_yiu_style_profile",
    "YIU_VISUAL_SYSTEM_NAME",
    "YiuViewStyleProfile",
]
