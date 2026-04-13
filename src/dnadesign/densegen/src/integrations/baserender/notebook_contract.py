"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/integrations/baserender/notebook_contract.py

DenseGen notebook-to-BaseRender contract for records rendering defaults.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from dnadesign.baserender import DENSEGEN_TFBS_REQUIRED_KEYS, cruncher_showcase_style_overrides

REQUIRED_ADAPTER_COLUMN_KEYS = ("id", "sequence", "annotations")
REQUIRED_TFBS_ENTRY_KEYS = DENSEGEN_TFBS_REQUIRED_KEYS

_NOTEBOOK_COLORBLIND_PASTEL_PALETTE: Mapping[str, str] = {
    "tf:background": "#C3CAD3",
    "tf:lexA": "#5DADE2",
    "tf:cpxR": "#2D9B66",
    "tf:baeR": "#E58A2B",
    "promoter:sigma70_core:upstream": "#7D86D1",
    "promoter:sigma70_core:downstream": "#C886D1",
}

_KNOWN_ACRONYMS = {
    "cbc": "CBC",
    "dna": "DNA",
    "fimo": "FIMO",
    "gurobi": "GUROBI",
    "pwm": "PWM",
    "tfbs": "TFBS",
    "usr": "USR",
}


@dataclass(frozen=True)
class DenseGenNotebookRenderContract:
    adapter_kind: str
    adapter_columns: Mapping[str, str]
    adapter_policies: Mapping[str, object]
    style_preset: str
    style_overrides: Mapping[str, object]
    record_window_limit: int


def format_densegen_workspace_heading(raw_name: str) -> str:
    text = str(raw_name or "").strip()
    if not text:
        return "DenseGen Workspace"
    tokens = [token for token in re.split(r"[_\-\s]+", text) if token]
    if not tokens:
        return "DenseGen Workspace"
    words: list[str] = []
    for token in tokens:
        normalized = token.strip()
        lower = normalized.lower()
        if lower in _KNOWN_ACRONYMS:
            words.append(_KNOWN_ACRONYMS[lower])
            continue
        words.append(normalized[:1].upper() + normalized[1:])
    return " ".join(words)


def format_densegen_record_display_id(record_id: str) -> str:
    record_text = str(record_id or "unknown")
    if len(record_text) > 16:
        return f"{record_text[:8]}...{record_text[-4:]}"
    return record_text


def format_densegen_plan_label(plan_name: str) -> str:
    raw_plan_text = str(plan_name or "").strip()
    if not raw_plan_text:
        return "unscoped"
    plan_base = raw_plan_text.split("__", 1)[0] if "__" in raw_plan_text else raw_plan_text
    plan_label = " ".join(str(plan_base).replace("_", " ").split())
    return plan_label or "unscoped"


def densegen_video_subtitle_text(*, record_id: str, plan_name: str) -> str:
    return f"Sequence {format_densegen_record_display_id(record_id)} | Plan {format_densegen_plan_label(plan_name)}"


def _validate_notebook_render_contract(contract: DenseGenNotebookRenderContract) -> None:
    if not isinstance(contract.adapter_kind, str) or not contract.adapter_kind.strip():
        raise ValueError("adapter_kind must be a non-empty string")
    if not isinstance(contract.style_preset, str) or not contract.style_preset.strip():
        raise ValueError("style_preset must be a non-empty string")
    if not isinstance(contract.style_overrides, Mapping):
        raise ValueError("style_overrides must be a mapping")
    missing_keys = [key for key in REQUIRED_ADAPTER_COLUMN_KEYS if key not in contract.adapter_columns]
    if missing_keys:
        raise ValueError(f"adapter_columns missing required keys: {missing_keys}")
    for key in REQUIRED_ADAPTER_COLUMN_KEYS:
        value = contract.adapter_columns[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"adapter_columns[{key!r}] must be a non-empty string")
    if int(contract.record_window_limit) <= 0:
        raise ValueError("record_window_limit must be > 0")


def densegen_notebook_render_contract() -> DenseGenNotebookRenderContract:
    showcase_style = dict(cruncher_showcase_style_overrides())
    showcase_palette = dict(showcase_style.get("palette") or {})
    showcase_palette.update(_NOTEBOOK_COLORBLIND_PASTEL_PALETTE)
    contract = DenseGenNotebookRenderContract(
        adapter_kind="densegen_tfbs",
        adapter_columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
        },
        adapter_policies={"on_invalid_row": "error"},
        style_preset="presentation_default",
        style_overrides={
            "palette": showcase_palette,
            "layout": {"outer_pad_cells": 0.62},
            "sequence": {"to_kmer_gap_cells": 0.38},
            "legend_pad_px": 24.0,
            "legend_patch_w": 28.0,
            "legend_patch_h": 13.0,
            "legend_font_size": 14,
            "legend_gap_patch_text": 11.0,
            "legend_gap_x": 44.0,
        },
        record_window_limit=500,
    )
    _validate_notebook_render_contract(contract)
    return contract
