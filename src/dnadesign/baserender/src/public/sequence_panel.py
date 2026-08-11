"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/public/sequence_panel.py

Public sequence-panel contract helpers for BaseRender.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..integrations.sequence_panels import (
    sequence_panel_defaults,
)
from ..integrations.styles import integration_style_overrides

BASERENDER_SEQUENCE_PANEL_CONTRACT_ID = "dnadesign.baserender.sequence_panel.v1"
BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION = "1"


@dataclass(frozen=True)
class SequencePanelConfig:
    adapter_kind: str
    adapter_columns: Mapping[str, object]
    adapter_policies: Mapping[str, object]
    style_profile: str
    style_preset: str | Path | None = "presentation_default"
    style_overrides: Mapping[str, object] | None = None
    renderer_name: str = "sequence_rows"
    alphabet: str = "DNA"
    target_width_px: int = 2200
    target_height_px: int = 430
    vertical_anchor: str = "center"
    canvas_top_pad_px: int = 0


@dataclass(frozen=True)
class SequencePanelDiagnostics:
    contract_id: str
    contract_version: str
    style_profile: str
    style_preset: str | None
    adapter_kind: str
    renderer_name: str
    sequence_length_bp: int
    feature_count: int
    strand_count: int
    legend_entries: tuple[str, ...]
    image_width_px: int
    image_height_px: int
    strand_center_y_px: float
    title: str | None
    record_label: str | None


@dataclass(frozen=True)
class SequencePanelImage:
    image: Any
    diagnostics: SequencePanelDiagnostics


def sequence_panel_config_for_adapter(
    adapter_kind: str,
    *,
    style_profile: str,
    adapter_columns: Mapping[str, object] | None = None,
    adapter_policies: Mapping[str, object] | None = None,
    style_overrides: Mapping[str, object] | None = None,
    target_width_px: int = 2200,
    target_height_px: int = 430,
    vertical_anchor: str = "center",
    canvas_top_pad_px: int = 0,
) -> SequencePanelConfig:
    profile_id = str(style_profile).strip()
    defaults = sequence_panel_defaults(adapter_kind, style_profile=profile_id)
    default_columns = dict(defaults.columns)
    default_policies = dict(defaults.policies)
    merged_policies = dict(default_policies)
    merged_policies.update(dict(adapter_policies or {}))
    merged_style = integration_style_overrides(profile_id)
    merged_style.update(dict(style_overrides or {}))
    return SequencePanelConfig(
        adapter_kind=str(adapter_kind),
        adapter_columns=dict(adapter_columns or default_columns),
        adapter_policies=merged_policies,
        style_profile=profile_id,
        style_preset="presentation_default",
        style_overrides=merged_style,
        target_width_px=int(target_width_px),
        target_height_px=int(target_height_px),
        vertical_anchor=str(vertical_anchor),
        canvas_top_pad_px=int(canvas_top_pad_px),
    )


__all__ = [
    "BASERENDER_SEQUENCE_PANEL_CONTRACT_ID",
    "BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION",
    "SequencePanelConfig",
    "SequencePanelDiagnostics",
    "SequencePanelImage",
    "sequence_panel_config_for_adapter",
]
