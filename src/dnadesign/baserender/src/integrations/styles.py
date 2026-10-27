"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/styles.py

Resolve style profiles contributed by built-in integrations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..core import SchemaError
from .registry import registered_style_profile


def integration_style_overrides(profile_name: str) -> dict[str, object]:
    name = str(profile_name).strip()
    descriptor = registered_style_profile(name)
    if descriptor is None:
        raise SchemaError(f"Unknown BaseRender style profile: {profile_name!r}")
    return descriptor.style_factory()


__all__ = ["integration_style_overrides"]
