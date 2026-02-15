"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/render/__init__.py

Lazy rendering exports to avoid eager matplotlib import in non-render code paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

__all__ = [
    "Renderer",
    "Palette",
    "LayoutContext",
    "compute_layout",
    "legend_entries_for_record",
    "get_renderer",
    "render_record",
]


def __getattr__(name: str):
    if name == "Palette":
        from .palette import Palette

        return Palette
    if name in {"Renderer", "get_renderer", "render_record"}:
        from .renderer import Renderer, get_renderer, render_record

        return {"Renderer": Renderer, "get_renderer": get_renderer, "render_record": render_record}[name]
    if name in {"LayoutContext", "compute_layout"}:
        from .layout import LayoutContext, compute_layout

        return {"LayoutContext": LayoutContext, "compute_layout": compute_layout}[name]
    if name == "legend_entries_for_record":
        from .legend import legend_entries_for_record

        return legend_entries_for_record
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
