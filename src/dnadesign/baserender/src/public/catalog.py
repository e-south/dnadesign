"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/public/catalog.py

Metadata-only BaseRender catalog queries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..adapters.registry import get_adapter_descriptor as _get_adapter_descriptor
from ..adapters.registry import list_adapter_descriptors
from ..config.job_contracts import render_contract_descriptor, render_contract_descriptors
from ..render.renderer import get_renderer_descriptor as _get_renderer_descriptor
from ..render.renderer import renderer_descriptors


def list_adapters() -> tuple[str, ...]:
    return tuple(descriptor.kind for descriptor in list_adapter_descriptors())


def get_adapter_descriptor(kind: str):
    return _get_adapter_descriptor(kind)


def list_renderers() -> tuple[str, ...]:
    return tuple(descriptor.name for descriptor in renderer_descriptors())


def get_renderer_descriptor(name: str):
    return _get_renderer_descriptor(name)


def list_render_contracts() -> tuple[str, ...]:
    return tuple(descriptor.kind for descriptor in render_contract_descriptors())


def get_render_contract_descriptor(kind: str):
    return render_contract_descriptor(kind)


__all__ = [
    "get_adapter_descriptor",
    "get_render_contract_descriptor",
    "get_renderer_descriptor",
    "list_adapters",
    "list_render_contracts",
    "list_renderers",
]
