"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/effects/registry.py

Effect drawing registry keyed by effect kind.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ...config import Style
from ...core import Effect, Record, RenderingError
from ..layout import LayoutContext
from ..palette import Palette

EffectDrawer = Callable[
    [object, Effect, Record, LayoutContext, Style, Palette, dict[str, tuple[float, float, float, float]]], None
]
EffectValidator = Callable[
    [Effect, Record, LayoutContext, Style, Palette, dict[str, tuple[float, float, float, float]]], None
]


@dataclass(frozen=True)
class _RegisteredEffect:
    drawer: EffectDrawer
    validator: EffectValidator


_EFFECTS: dict[str, _RegisteredEffect] = {}


def clear_effect_drawers() -> None:
    _EFFECTS.clear()


def register_effect_drawer(kind: str, drawer: EffectDrawer, *, validator: EffectValidator) -> None:
    _EFFECTS[kind] = _RegisteredEffect(drawer=drawer, validator=validator)


def get_effect_drawer(kind: str) -> EffectDrawer:
    registered = _EFFECTS.get(kind)
    if registered is None:
        raise RenderingError(f"Unknown effect kind: {kind}")
    return registered.drawer


def validate_effect(
    effect: Effect,
    record: Record,
    layout: LayoutContext,
    style: Style,
    palette: Palette,
    feature_boxes: dict[str, tuple[float, float, float, float]],
) -> None:
    registered = _EFFECTS.get(effect.kind)
    if registered is None:
        raise RenderingError(f"Unknown effect kind: {effect.kind}")
    registered.validator(effect, record, layout, style, palette, feature_boxes)


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
