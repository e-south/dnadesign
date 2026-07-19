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

from ..core import SchemaError, ensure
from ..styles.curated import cruncher_showcase_style_overrides as _cruncher_showcase_style_overrides

BASERENDER_SEQUENCE_PANEL_CONTRACT_ID = "dnadesign.baserender.sequence_panel.v1"
BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION = "1"
DEFAULT_SEQUENCE_PANEL_PROFILE = "promoter_compact_slide.v1"


@dataclass(frozen=True)
class SequencePanelConfig:
    adapter_kind: str
    adapter_columns: Mapping[str, object]
    adapter_policies: Mapping[str, object]
    style_profile: str = DEFAULT_SEQUENCE_PANEL_PROFILE
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


def _densegen_tfbs_adapter_defaults() -> tuple[dict[str, str], dict[str, object]]:
    return (
        {
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
        },
        {"on_invalid_row": "error", "require_non_empty": False},
    )


def _usr_genbank_adapter_defaults() -> tuple[dict[str, str], dict[str, object]]:
    return (
        {
            "sequence": "sequence",
            "annotations": "seq_annot__features",
            "id": "id",
            "overlay_text": "usr_label__primary",
            "source_file": "seq_annot__source_file",
            "product_kind": "derived__product_kind",
        },
        {"overlay_text_template": "{overlay_text}", "on_invalid_row": "error"},
    )


def _sequence_panel_profile_style_overrides(profile: str) -> dict[str, object]:
    profile_id = str(profile).strip()
    ensure(profile_id == DEFAULT_SEQUENCE_PANEL_PROFILE, f"Unknown sequence panel profile: {profile!r}", SchemaError)
    base = dict(_cruncher_showcase_style_overrides())
    palette = dict(base.get("palette") or {})
    palette.update(
        {
            "tf:background": "#C3CAD3",
            "tf:lexA": "#5DADE2",
            "tf:cpxR": "#2D9B66",
            "tf:baeR": "#E58A2B",
            "promoter:sigma70_core:upstream": "#7D86D1",
            "promoter:sigma70_core:downstream": "#C886D1",
        }
    )
    base["palette"] = palette
    base["layout"] = {**dict(base.get("layout") or {}), "outer_pad_cells": 0.62}
    base["sequence"] = {**dict(base.get("sequence") or {}), "to_kmer_gap_cells": 0.38}
    base.update(
        {
            "legend": True,
            "legend_mode": "bottom",
            "legend_height_px": 136.0,
            "legend_pad_px": 36.0,
            "legend_content_gap_px": 18.0,
            "legend_patch_w": 88.0,
            "legend_patch_h": 34.0,
            "legend_font_size": 24,
            "legend_gap_patch_text": 22.0,
            "legend_gap_x": 44.0,
            "legend_vertical_align": 1.0,
            "show_reverse_complement": True,
            "font_size_seq": 24,
            "font_size_label": 24,
            "font_size_feature_label": 24,
            "font_size_annotation_label": 24,
            "font_size_span_link_label": 24,
            "span_link_line_width": 3.2,
            "span_link_tick_line_width": 2.8,
            "uniform_display_font_size": True,
            "overlay_vertical_anchor": "content_top",
        }
    )
    return base


def _adapter_defaults(adapter_kind: str) -> tuple[dict[str, str], dict[str, object]]:
    kind = str(adapter_kind).strip()
    if kind == "densegen_tfbs":
        return _densegen_tfbs_adapter_defaults()
    if kind == "usr_genbank_annotations_v1":
        return _usr_genbank_adapter_defaults()
    raise SchemaError(f"Unsupported sequence panel adapter kind: {adapter_kind!r}")


def sequence_panel_config_for_adapter(
    adapter_kind: str,
    *,
    style_profile: str = DEFAULT_SEQUENCE_PANEL_PROFILE,
    adapter_columns: Mapping[str, object] | None = None,
    adapter_policies: Mapping[str, object] | None = None,
    style_overrides: Mapping[str, object] | None = None,
    target_width_px: int = 2200,
    target_height_px: int = 430,
    vertical_anchor: str = "center",
    canvas_top_pad_px: int = 0,
) -> SequencePanelConfig:
    default_columns, default_policies = _adapter_defaults(adapter_kind)
    merged_policies = dict(default_policies)
    merged_policies.update(dict(adapter_policies or {}))
    merged_style = _sequence_panel_profile_style_overrides(style_profile)
    merged_style.update(dict(style_overrides or {}))
    return SequencePanelConfig(
        adapter_kind=str(adapter_kind),
        adapter_columns=dict(adapter_columns or default_columns),
        adapter_policies=merged_policies,
        style_profile=str(style_profile),
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
    "DEFAULT_SEQUENCE_PANEL_PROFILE",
    "SequencePanelConfig",
    "SequencePanelDiagnostics",
    "SequencePanelImage",
    "sequence_panel_config_for_adapter",
]
