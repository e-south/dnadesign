"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/presets/runtime.py

Runtime helpers for resolving cluster presets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import typer

from .loader import load_all as load_presets


def apply_preset(kind: str, preset_name: str | None) -> dict[str, Any]:
    if not preset_name:
        return {}
    preset = load_presets().get(preset_name)
    if preset is None:
        raise typer.BadParameter(f"Preset '{preset_name}' not found.")
    if preset.kind != kind:
        raise typer.BadParameter(
            f"Preset '{preset_name}' is kind='{preset.kind}', but this command expects kind='{kind}'."
        )
    return preset.params or {}


def apply_plot_preset(preset_name: str | None) -> dict[str, Any]:
    if not preset_name:
        return {}
    preset = load_presets().get(preset_name)
    return (preset.plot or {}) if preset else {}


__all__ = ["apply_plot_preset", "apply_preset"]
