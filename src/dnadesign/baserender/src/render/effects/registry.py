"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/render/effects/registry.py

Effect drawing registry keyed by effect kind.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable

from ...config import Style
from ...core import Effect, Record, RenderingError
from ..layout import LayoutContext
from ..palette import Palette

EffectDrawer = Callable[
    [object, Effect, Record, LayoutContext, Style, Palette, dict[str, tuple[float, float, float, float]]], None
]

_EFFECT_DRAWERS: dict[str, EffectDrawer] = {}


def clear_effect_drawers() -> None:
    _EFFECT_DRAWERS.clear()


def register_effect_drawer(kind: str, drawer: EffectDrawer) -> None:
    _EFFECT_DRAWERS[kind] = drawer


def get_effect_drawer(kind: str) -> EffectDrawer:
    drawer = _EFFECT_DRAWERS.get(kind)
    if drawer is None:
        raise RenderingError(f"Unknown effect kind: {kind}")
    return drawer


def draw_effect(
    ax,
    effect: Effect,
    record: Record,
    layout: LayoutContext,
    style: Style,
    palette: Palette,
    feature_boxes: dict[str, tuple[float, float, float, float]],
) -> None:
    drawer = get_effect_drawer(effect.kind)
    drawer(ax, effect, record, layout, style, palette, feature_boxes)
