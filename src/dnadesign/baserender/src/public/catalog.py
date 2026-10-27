"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/public/catalog.py

Metadata-only BaseRender catalog queries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..config.job_contracts import render_contract_descriptor, render_contract_descriptors
from ..integrations import adapter_descriptor as _get_adapter_descriptor
from ..integrations import adapter_descriptors, transform_descriptor, transform_descriptors
from ..integrations.registry import registered_style_profile, registered_style_profiles
from ..render.renderer import get_renderer_descriptor as _get_renderer_descriptor
from ..render.renderer import renderer_descriptors


def list_adapters() -> tuple[str, ...]:
    return tuple(descriptor.kind for descriptor in adapter_descriptors())


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


def list_transforms() -> tuple[str, ...]:
    return tuple(descriptor.name for descriptor in transform_descriptors())


def get_transform_descriptor(name: str):
    return transform_descriptor(name)


def list_style_profiles() -> tuple[str, ...]:
    return tuple(descriptor.name for descriptor in registered_style_profiles())


def get_style_profile_descriptor(name: str):
    descriptor = registered_style_profile(str(name).strip())
    if descriptor is None:
        from ..core import SchemaError

        raise SchemaError(f"Unknown BaseRender style profile: {name!r}")
    return descriptor


__all__ = [
    "get_adapter_descriptor",
    "get_render_contract_descriptor",
    "get_renderer_descriptor",
    "get_style_profile_descriptor",
    "get_transform_descriptor",
    "list_adapters",
    "list_render_contracts",
    "list_renderers",
    "list_style_profiles",
    "list_transforms",
]
