"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/render/effects/__init__.py

Effect drawer exports and registration helper.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .motif_logo import draw_motif_logo
from .registry import clear_effect_drawers, draw_effect, get_effect_drawer, register_effect_drawer
from .span_link import draw_span_link


def register_builtin_effect_drawers() -> None:
    register_effect_drawer("span_link", draw_span_link)
    register_effect_drawer("motif_logo", draw_motif_logo)


__all__ = [
    "draw_effect",
    "clear_effect_drawers",
    "register_effect_drawer",
    "get_effect_drawer",
    "draw_span_link",
    "draw_motif_logo",
    "register_builtin_effect_drawers",
]
