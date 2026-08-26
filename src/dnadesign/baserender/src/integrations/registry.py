"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/registry.py

Compose the built-in BaseRender integration registry.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..core import RenderContractDescriptor
from .cassette import PROVIDER as CASSETTE
from .contracts import (
    AdapterDescriptor,
    IntegrationProvider,
    SequencePanelDefaults,
    StyleProfileDescriptor,
    TransformDescriptor,
)
from .cruncher import PROVIDER as CRUNCHER
from .densegen import PROVIDER as DENSEGEN
from .generic import PROVIDER as GENERIC
from .junction import PROVIDER as JUNCTION
from .motif_annotation import PROVIDER as MOTIF_ANNOTATION
from .promoter_panel import PROVIDER as PROMOTER_PANEL
from .usr import PROVIDER as USR
from .yiu import PROVIDER as YIU

_PROVIDERS = (
    CASSETTE,
    CRUNCHER,
    DENSEGEN,
    GENERIC,
    JUNCTION,
    MOTIF_ANNOTATION,
    PROMOTER_PANEL,
    USR,
    YIU,
)

_PROVIDER_NAMES = tuple(provider.name for provider in _PROVIDERS)
if len(set(_PROVIDER_NAMES)) != len(_PROVIDER_NAMES):
    duplicates = sorted(name for name in set(_PROVIDER_NAMES) if _PROVIDER_NAMES.count(name) > 1)
    raise RuntimeError(f"Duplicate BaseRender integration providers: {duplicates}")


def _unique_by_name(items, *, item_name: str, name_getter):
    out = {}
    for provider in _PROVIDERS:
        for item in items(provider):
            name = name_getter(item)
            if name in out:
                raise RuntimeError(f"Duplicate BaseRender {item_name} descriptor: {name}")
            out[name] = item
    return out


_ADAPTERS: dict[str, AdapterDescriptor] = _unique_by_name(
    lambda provider: provider.adapters,
    item_name="adapter",
    name_getter=lambda descriptor: descriptor.kind,
)
_TRANSFORMS: dict[str, TransformDescriptor] = _unique_by_name(
    lambda provider: provider.transforms,
    item_name="transform",
    name_getter=lambda descriptor: descriptor.name,
)
_SEQUENCE_PANELS: dict[str, SequencePanelDefaults] = _unique_by_name(
    lambda provider: provider.sequence_panels,
    item_name="sequence-panel",
    name_getter=lambda descriptor: descriptor.adapter_kind,
)
_STYLE_PROFILES: dict[str, StyleProfileDescriptor] = _unique_by_name(
    lambda provider: provider.style_profiles,
    item_name="style-profile",
    name_getter=lambda descriptor: descriptor.name,
)
_RENDER_CONTRACTS: dict[str, RenderContractDescriptor] = _unique_by_name(
    lambda provider: provider.render_contracts,
    item_name="render-contract",
    name_getter=lambda descriptor: descriptor.kind,
)


def integration_providers() -> tuple[IntegrationProvider, ...]:
    return tuple(sorted(_PROVIDERS, key=lambda provider: provider.name))


def registered_adapters() -> tuple[AdapterDescriptor, ...]:
    return tuple(_ADAPTERS[name] for name in sorted(_ADAPTERS))


def registered_transforms() -> tuple[TransformDescriptor, ...]:
    return tuple(_TRANSFORMS[name] for name in sorted(_TRANSFORMS))


def registered_adapter(kind: str) -> AdapterDescriptor | None:
    return _ADAPTERS.get(kind)


def registered_transform(name: str) -> TransformDescriptor | None:
    return _TRANSFORMS.get(name)


def registered_sequence_panel(adapter_kind: str) -> SequencePanelDefaults | None:
    return _SEQUENCE_PANELS.get(adapter_kind)


def registered_style_profiles() -> tuple[StyleProfileDescriptor, ...]:
    return tuple(_STYLE_PROFILES[name] for name in sorted(_STYLE_PROFILES))


def registered_style_profile(name: str) -> StyleProfileDescriptor | None:
    return _STYLE_PROFILES.get(name)


def registered_render_contracts() -> tuple[RenderContractDescriptor, ...]:
    return tuple(_RENDER_CONTRACTS[name] for name in sorted(_RENDER_CONTRACTS))


__all__ = [
    "integration_providers",
    "registered_adapter",
    "registered_adapters",
    "registered_sequence_panel",
    "registered_style_profile",
    "registered_style_profiles",
    "registered_render_contracts",
    "registered_transform",
    "registered_transforms",
]
