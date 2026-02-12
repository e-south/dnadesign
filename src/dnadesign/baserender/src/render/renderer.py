"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/render/renderer.py

Renderer registry and pre-render contract enforcement for Record v1.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..config import Style
from ..core import ContractError, Record, RenderingError, validate_record_kinds
from .palette import Palette
from .sequence_rows import SequenceRowsRenderer


class Renderer(Protocol):
    def render(self, record: Record, style: Style, palette: Palette): ...


@dataclass(frozen=True)
class _RendererRegistry:
    renderers: dict[str, Renderer]

    def get(self, name: str) -> Renderer:
        renderer = self.renderers.get(name)
        if renderer is None:
            raise RenderingError(f"Unknown renderer: {name}")
        return renderer


_REGISTRY = _RendererRegistry(renderers={"sequence_rows": SequenceRowsRenderer()})


def get_renderer(name: str) -> Renderer:
    return _REGISTRY.get(name)


def render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
    try:
        validated = record.validate()
        validate_record_kinds(validated)
    except ContractError as exc:
        raise RenderingError(str(exc)) from exc

    renderer = get_renderer(renderer_name)
    return renderer.render(validated, style, palette)
