"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/composition/review_manifest.py

Composition-review manifest construction and bundle index updates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.contracts.visual import CompositionReviewSvgV1, SequenceEvidenceMapV1

from ..contracts.errors import ValidationError
from .review_assets import (
    COMPONENT_PANEL_EMPHASIS,
    COMPONENT_SOURCE_TITLE_POLICY,
    COMPONENT_SPAN_SVG_PATH,
    COMPOSITION_REVIEW_MANIFEST_PATH,
    COMPOSITION_REVIEW_PNG_PATH,
    COMPOSITION_REVIEW_SVG_PATH,
    REVIEW_PNG_PPI,
    REVIEW_PNG_SCALE,
    SEQUENCE_EVIDENCE_MAP_PATH,
    STRUCTURE_FIT_POLICY,
    STRUCTURE_SVG_PATH,
    composition_id_from_visual_contract,
    write_json,
)


def build_review_manifest(
    *,
    bundle: Path,
    visual_contract: SequenceEvidenceMapV1,
    target_nucleotide_font_size_px: float,
    structure_width_px: float,
    component_width_px: float,
    structure_scale: float,
    component_scale: float,
    structure_effective_font_size: float,
    component_effective_font_size: float,
    component_source_title_omitted_count: int,
) -> CompositionReviewSvgV1:
    composition_id = composition_id_from_visual_contract(visual_contract)
    width_ratio = structure_width_px / component_width_px
    return CompositionReviewSvgV1(
        review_id=f"{composition_id}.composition_review",
        composition_id=composition_id,
        sequence_id=visual_contract.state_id,
        sequence_sha256=str(visual_contract.meta.get("sequence_sha256") or ""),
        length=len(visual_contract.primary_sequence),
        sources={
            "structure_svg": STRUCTURE_SVG_PATH.as_posix(),
            "component_span_svg": COMPONENT_SPAN_SVG_PATH.as_posix(),
            "visual_contract": SEQUENCE_EVIDENCE_MAP_PATH.as_posix(),
            "bundle_manifest": "manifest.json" if (bundle / "manifest.json").is_file() else None,
        },
        artifacts={
            "review_svg": COMPOSITION_REVIEW_SVG_PATH.as_posix(),
            "review_png": COMPOSITION_REVIEW_PNG_PATH.as_posix(),
        },
        layout={
            "row_count": 2,
            "panel_order": ["secondary_structure", "component_span"],
            "component_nucleotide_font_size_px": target_nucleotide_font_size_px,
            "structure_fit_policy": STRUCTURE_FIT_POLICY,
            "structure_scale": structure_scale,
            "component_scale": component_scale,
            "structure_width_px": structure_width_px,
            "component_width_px": component_width_px,
            "structure_effective_nucleotide_font_size_px": structure_effective_font_size,
            "component_effective_nucleotide_font_size_px": component_effective_font_size,
            "component_panel_emphasis": COMPONENT_PANEL_EMPHASIS,
            "component_source_title_policy": COMPONENT_SOURCE_TITLE_POLICY,
            "structure_to_component_width_ratio": width_ratio,
            "vertical_gap_px": 18.0,
            "review_png_scale": REVIEW_PNG_SCALE,
            "review_png_ppi": REVIEW_PNG_PPI,
        },
        qa={
            "subplot_visual_weight_balanced": (
                0.75 <= width_ratio <= 0.9 and component_effective_font_size >= target_nucleotide_font_size_px * 1.45
            ),
            "component_panel_emphasis_applied": True,
            "component_source_title_omitted": component_source_title_omitted_count > 0,
            "component_source_title_omitted_count": component_source_title_omitted_count,
            "warnings": [],
            "errors": [],
        },
    )


def update_bundle_manifest(bundle: Path, review_manifest: CompositionReviewSvgV1) -> None:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Bundle manifest is not valid JSON: {manifest_path}") from exc
    artifacts = payload.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValidationError("Bundle manifest artifacts must be an object.")
    artifacts["composition_review"] = COMPOSITION_REVIEW_MANIFEST_PATH.as_posix()
    artifacts["composition_review_svg"] = review_manifest.artifacts.review_svg
    artifacts["composition_review_png"] = review_manifest.artifacts.review_png
    write_json(manifest_path, payload)


__all__ = [
    "build_review_manifest",
    "update_bundle_manifest",
]
