"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/sequence_panels.py

Resolve optional sequence-panel profiles from integration descriptors.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..core import SchemaError
from .contracts import SequencePanelDefaults
from .registry import registered_sequence_panel


def sequence_panel_defaults(adapter_kind: str, *, style_profile: str) -> SequencePanelDefaults:
    descriptor = registered_sequence_panel(str(adapter_kind).strip())
    if descriptor is None:
        raise SchemaError(f"Unsupported sequence panel adapter kind: {adapter_kind!r}")
    if style_profile not in descriptor.supported_profiles:
        allowed = ", ".join(descriptor.supported_profiles)
        raise SchemaError(
            f"Unknown sequence panel profile: {style_profile!r}; supported profiles for {adapter_kind!r}: {allowed}"
        )
    return descriptor


__all__ = [
    "sequence_panel_defaults",
]
