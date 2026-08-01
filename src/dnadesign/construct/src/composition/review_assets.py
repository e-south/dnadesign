"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/composition/review_assets.py

Composition-review source assets and artifact writers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from dnadesign.contracts.visual import SequenceEvidenceMapV1

from ..contracts.errors import ValidationError
from .svg_geometry import extract_nucleotide_font_size_px, viewbox
from .visual import SEQUENCE_EVIDENCE_MAP_PATH

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

STRUCTURE_SVG_PATH = Path("visual/viennarna_secondary_structure/secondary_structure.annotated.svg")
COMPONENT_SPAN_SVG_PATH = Path("visual/renders/component_span_qa_svg.render-v1/component_span_qa.svg")
COMPOSITION_REVIEW_DIR = Path("visual/reviews")
COMPOSITION_REVIEW_SVG_PATH = COMPOSITION_REVIEW_DIR / "composition_overview.svg"
COMPOSITION_REVIEW_PNG_PATH = COMPOSITION_REVIEW_DIR / "composition_overview.png"
COMPOSITION_REVIEW_MANIFEST_PATH = COMPOSITION_REVIEW_DIR / "composition_review_svg_v1.json"

STRUCTURE_PANEL_WIDTH_RATIO = 1.25
COMPONENT_TO_STRUCTURE_REVIEW_WIDTH_RATIO = 1.22
STRUCTURE_FIT_POLICY = "balanced_visual_weight"
COMPONENT_PANEL_EMPHASIS = "filled_region_plain_glyph_review"
COMPONENT_SOURCE_TITLE_POLICY = "omit_redundant_source_title"
COMPONENT_SOURCE_TITLE_COLOR = "#6b7280"
REVIEW_PNG_SCALE = 3.0
REVIEW_PNG_PPI = 216.0


@dataclass(frozen=True)
class SvgAsset:
    path: Path
    root: ET.Element
    viewbox: tuple[float, float, float, float]
    source_nucleotide_font_size_px: float

    @property
    def width(self) -> float:
        return self.viewbox[2]

    @property
    def height(self) -> float:
        return self.viewbox[3]


def load_visual_contract(path: Path) -> SequenceEvidenceMapV1:
    if not path.is_file():
        raise ValidationError(f"Visual contract not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Visual contract is not valid JSON: {path}") from exc
    try:
        return SequenceEvidenceMapV1.model_validate(payload)
    except ValueError as exc:
        raise ValidationError(f"Visual contract failed validation: {path}: {exc}") from exc


def load_svg_asset(path: Path, *, default_nucleotide_font_size_px: float) -> SvgAsset:
    if not path.is_file():
        raise ValidationError(f"Review source SVG not found: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValidationError(f"Review source SVG is not parseable XML: {path}") from exc
    svg_viewbox = viewbox(root)
    if svg_viewbox[2] <= 0 or svg_viewbox[3] <= 0:
        raise ValidationError(f"Review source SVG has nonpositive dimensions: {path}")
    font_size = extract_nucleotide_font_size_px(root, default=default_nucleotide_font_size_px)
    if font_size <= 0:
        raise ValidationError(f"Review source SVG has nonpositive nucleotide font size: {path}")
    return SvgAsset(
        path=path,
        root=root,
        viewbox=svg_viewbox,
        source_nucleotide_font_size_px=font_size,
    )


def composition_id_from_visual_contract(visual_contract: SequenceEvidenceMapV1) -> str:
    source_id = visual_contract.meta.get("visual_scope", {}).get("source_sequence_id")
    if isinstance(source_id, str) and source_id.strip():
        return source_id.strip()
    return visual_contract.state_id.rsplit(".", maxsplit=1)[0]


def write_review_png(source_svg: Path, output_png: Path) -> None:
    try:
        import vl_convert as vlc
    except ImportError as exc:  # pragma: no cover - dependency is pinned in the managed environment.
        raise ValidationError("vl-convert-python is required to publish composition review PNG artifacts.") from exc
    try:
        png_bytes = vlc.svg_to_png(
            source_svg.read_text(encoding="utf-8"),
            scale=REVIEW_PNG_SCALE,
            ppi=REVIEW_PNG_PPI,
        )
    except Exception as exc:  # pragma: no cover - renderer-specific failure details.
        raise ValidationError(f"Failed to rasterize composition review SVG to PNG: {source_svg}") from exc
    if not bytes(png_bytes).startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValidationError(f"Composition review PNG renderer returned invalid PNG data for: {source_svg}")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_png.write_bytes(bytes(png_bytes))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "COMPONENT_PANEL_EMPHASIS",
    "COMPONENT_SOURCE_TITLE_COLOR",
    "COMPONENT_SOURCE_TITLE_POLICY",
    "COMPONENT_SPAN_SVG_PATH",
    "COMPONENT_TO_STRUCTURE_REVIEW_WIDTH_RATIO",
    "COMPOSITION_REVIEW_MANIFEST_PATH",
    "COMPOSITION_REVIEW_PNG_PATH",
    "COMPOSITION_REVIEW_SVG_PATH",
    "REVIEW_PNG_PPI",
    "REVIEW_PNG_SCALE",
    "SEQUENCE_EVIDENCE_MAP_PATH",
    "STRUCTURE_FIT_POLICY",
    "STRUCTURE_PANEL_WIDTH_RATIO",
    "STRUCTURE_SVG_PATH",
    "SVG_NS",
    "XLINK_NS",
    "SvgAsset",
    "composition_id_from_visual_contract",
    "load_svg_asset",
    "load_visual_contract",
    "write_json",
    "write_review_png",
]
