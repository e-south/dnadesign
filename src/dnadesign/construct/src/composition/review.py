"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/composition/review.py

Composition-level visual review publisher for generated linear ssDNA bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from dnadesign.contracts.visual import CompositionReviewSvgV1

from ..contracts.errors import ValidationError
from .review_assets import (
    COMPONENT_SPAN_SVG_PATH,
    COMPONENT_TO_STRUCTURE_REVIEW_WIDTH_RATIO,
    COMPOSITION_REVIEW_MANIFEST_PATH,
    COMPOSITION_REVIEW_PNG_PATH,
    COMPOSITION_REVIEW_SVG_PATH,
    SEQUENCE_EVIDENCE_MAP_PATH,
    STRUCTURE_PANEL_WIDTH_RATIO,
    STRUCTURE_SVG_PATH,
    SVG_NS,
    XLINK_NS,
    load_svg_asset,
    load_visual_contract,
    write_json,
    write_review_png,
)
from .review_manifest import build_review_manifest, update_bundle_manifest
from .review_svg import compose_review_svg, validate_structure_nucleotide_text


def publish_composition_review_svg(
    artifact_bundle: str | Path,
    *,
    target_nucleotide_font_size_px: float = 6.0,
) -> CompositionReviewSvgV1:
    if target_nucleotide_font_size_px <= 0:
        raise ValidationError("target_nucleotide_font_size_px must be > 0.")
    bundle = Path(artifact_bundle).expanduser().resolve()
    visual_contract = load_visual_contract(bundle / SEQUENCE_EVIDENCE_MAP_PATH)
    structure = load_svg_asset(
        bundle / STRUCTURE_SVG_PATH,
        default_nucleotide_font_size_px=12.0,
    )
    component = load_svg_asset(
        bundle / COMPONENT_SPAN_SVG_PATH,
        default_nucleotide_font_size_px=target_nucleotide_font_size_px,
    )

    base_component_scale = target_nucleotide_font_size_px / component.source_nucleotide_font_size_px
    structure_width = component.width * base_component_scale * STRUCTURE_PANEL_WIDTH_RATIO
    component_width = structure_width * COMPONENT_TO_STRUCTURE_REVIEW_WIDTH_RATIO
    component_scale = component_width / component.width
    structure_scale = structure_width / structure.width
    component_width = component.width * component_scale
    structure_width = structure.width * structure_scale
    structure_effective_font_size = structure.source_nucleotide_font_size_px * structure_scale
    component_effective_font_size = component.source_nucleotide_font_size_px * component_scale
    svg_build = compose_review_svg(
        structure=structure,
        component=component,
        visual_contract=visual_contract,
        target_nucleotide_font_size_px=target_nucleotide_font_size_px,
        structure_scale=structure_scale,
        component_scale=component_scale,
        structure_effective_font_size=structure_effective_font_size,
        component_effective_font_size=component_effective_font_size,
    )
    validate_structure_nucleotide_text(
        svg_build.root,
        expected_count=len(visual_contract.primary_sequence),
        source_nucleotide_font_size_px=structure.source_nucleotide_font_size_px,
    )

    output_path = bundle / COMPOSITION_REVIEW_SVG_PATH
    manifest_path = bundle / COMPOSITION_REVIEW_MANIFEST_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    ET.ElementTree(svg_build.root).write(output_path, encoding="utf-8", xml_declaration=True)
    write_review_png(output_path, bundle / COMPOSITION_REVIEW_PNG_PATH)
    manifest = build_review_manifest(
        bundle=bundle,
        visual_contract=visual_contract,
        target_nucleotide_font_size_px=target_nucleotide_font_size_px,
        structure_width_px=structure_width,
        component_width_px=component_width,
        structure_scale=structure_scale,
        component_scale=component_scale,
        structure_effective_font_size=structure_effective_font_size,
        component_effective_font_size=component_effective_font_size,
        component_source_title_omitted_count=svg_build.component_source_title_omitted_count,
    )
    write_json(manifest_path, manifest.model_dump(mode="json"))
    write_json(
        bundle / "manifest" / "reviews" / "composition_review_svg_v1.json",
        manifest.model_dump(mode="json"),
    )
    update_bundle_manifest(bundle, manifest)
    return manifest


__all__ = [
    "COMPONENT_SPAN_SVG_PATH",
    "COMPOSITION_REVIEW_MANIFEST_PATH",
    "COMPOSITION_REVIEW_PNG_PATH",
    "COMPOSITION_REVIEW_SVG_PATH",
    "STRUCTURE_SVG_PATH",
    "publish_composition_review_svg",
]
