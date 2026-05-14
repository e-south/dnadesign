"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/composition_review.py

Composition-level visual review publisher for generated linear ssDNA bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from dnadesign.contracts.visual import CompositionReviewSvgV1, SequenceEvidenceMapV1

from .composition_visual import SEQUENCE_EVIDENCE_MAP_PATH
from .errors import ValidationError

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

STRUCTURE_SVG_PATH = Path("visual/viennarna_secondary_structure/secondary_structure.annotated.svg")
COMPONENT_SPAN_SVG_PATH = Path("visual/renders/component_span_qa_svg/component_span_qa.svg")
COMPOSITION_REVIEW_DIR = Path("visual/reviews")
COMPOSITION_REVIEW_SVG_PATH = COMPOSITION_REVIEW_DIR / "composition_overview.svg"
COMPOSITION_REVIEW_MANIFEST_PATH = COMPOSITION_REVIEW_DIR / "composition_review_svg_v1.json"
_STRUCTURE_PANEL_WIDTH_RATIO = 1.25
_COMPONENT_TO_STRUCTURE_REVIEW_WIDTH_RATIO = 1.22
_STRUCTURE_FIT_POLICY = "balanced_visual_weight"
_COMPONENT_PANEL_EMPHASIS = "bold_glyph_review"
_COMPONENT_SOURCE_TITLE_POLICY = "omit_redundant_source_title"
_COMPONENT_SOURCE_TITLE_COLOR = "#6b7280"


@dataclass(frozen=True)
class _ReviewSvgBuild:
    root: ET.Element
    component_source_title_omitted_count: int


@dataclass(frozen=True)
class _SvgAsset:
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


def publish_composition_review_svg(
    artifact_bundle: str | Path,
    *,
    target_nucleotide_font_size_px: float = 6.0,
) -> CompositionReviewSvgV1:
    if target_nucleotide_font_size_px <= 0:
        raise ValidationError("target_nucleotide_font_size_px must be > 0.")
    bundle = Path(artifact_bundle).expanduser().resolve()
    visual_contract = _load_visual_contract(bundle / SEQUENCE_EVIDENCE_MAP_PATH)
    structure = _load_svg_asset(
        bundle / STRUCTURE_SVG_PATH,
        default_nucleotide_font_size_px=12.0,
    )
    component = _load_svg_asset(
        bundle / COMPONENT_SPAN_SVG_PATH,
        default_nucleotide_font_size_px=target_nucleotide_font_size_px,
    )

    base_component_scale = target_nucleotide_font_size_px / component.source_nucleotide_font_size_px
    structure_width = component.width * base_component_scale * _STRUCTURE_PANEL_WIDTH_RATIO
    component_width = structure_width * _COMPONENT_TO_STRUCTURE_REVIEW_WIDTH_RATIO
    component_scale = component_width / component.width
    structure_scale = structure_width / structure.width
    component_width = component.width * component_scale
    structure_width = structure.width * structure_scale
    structure_effective_font_size = structure.source_nucleotide_font_size_px * structure_scale
    component_effective_font_size = component.source_nucleotide_font_size_px * component_scale
    svg_build = _compose_review_svg(
        structure=structure,
        component=component,
        composition_id=_composition_id_from_visual_contract(visual_contract),
        target_nucleotide_font_size_px=target_nucleotide_font_size_px,
        structure_scale=structure_scale,
        component_scale=component_scale,
        structure_effective_font_size=structure_effective_font_size,
        component_effective_font_size=component_effective_font_size,
    )

    output_path = bundle / COMPOSITION_REVIEW_SVG_PATH
    manifest_path = bundle / COMPOSITION_REVIEW_MANIFEST_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    ET.ElementTree(svg_build.root).write(output_path, encoding="utf-8", xml_declaration=True)
    manifest = _manifest(
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
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    _update_bundle_manifest(bundle, manifest)
    return manifest


def _load_visual_contract(path: Path) -> SequenceEvidenceMapV1:
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


def _load_svg_asset(path: Path, *, default_nucleotide_font_size_px: float) -> _SvgAsset:
    if not path.is_file():
        raise ValidationError(f"Review source SVG not found: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValidationError(f"Review source SVG is not parseable XML: {path}") from exc
    viewbox = _viewbox(root)
    if viewbox[2] <= 0 or viewbox[3] <= 0:
        raise ValidationError(f"Review source SVG has nonpositive dimensions: {path}")
    font_size = _extract_nucleotide_font_size_px(root, default=default_nucleotide_font_size_px)
    if font_size <= 0:
        raise ValidationError(f"Review source SVG has nonpositive nucleotide font size: {path}")
    return _SvgAsset(
        path=path,
        root=root,
        viewbox=viewbox,
        source_nucleotide_font_size_px=font_size,
    )


def _compose_review_svg(
    *,
    structure: _SvgAsset,
    component: _SvgAsset,
    composition_id: str,
    target_nucleotide_font_size_px: float,
    structure_scale: float,
    component_scale: float,
    structure_effective_font_size: float,
    component_effective_font_size: float,
) -> _ReviewSvgBuild:
    pad = 18.0
    gap = 18.0
    structure_width = structure.width * structure_scale
    structure_height = structure.height * structure_scale
    component_width = component.width * component_scale
    component_height = component.height * component_scale
    content_width = max(structure_width, component_width)
    outer_width = content_width + 2 * pad
    outer_height = structure_height + gap + component_height + 2 * pad
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": f"{outer_width:.0f}",
            "height": f"{outer_height:.0f}",
            "viewBox": f"0 0 {outer_width:.3f} {outer_height:.3f}",
            "data-dnadesign-contract-kind": "composition_review_svg_v1",
            "data-dnadesign-composition-id": composition_id,
            "data-dnadesign-component-nucleotide-font-size-px": f"{target_nucleotide_font_size_px:.3f}",
            "data-dnadesign-structure-effective-nucleotide-font-size-px": f"{structure_effective_font_size:.3f}",
            "data-dnadesign-component-effective-nucleotide-font-size-px": f"{component_effective_font_size:.3f}",
            "data-dnadesign-component-panel-emphasis": _COMPONENT_PANEL_EMPHASIS,
            "data-dnadesign-structure-fit-policy": _STRUCTURE_FIT_POLICY,
        },
    )
    title = ET.SubElement(root, f"{{{SVG_NS}}}title")
    title.text = f"{composition_id} composition review"
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}rect",
        {
            "x": "0",
            "y": "0",
            "width": f"{outer_width:.3f}",
            "height": f"{outer_height:.3f}",
            "style": "fill: #FFFFFF; stroke: none;",
        },
    )
    _append_panel(
        root,
        asset=structure,
        panel="secondary_structure",
        row=1,
        x=pad + (content_width - structure_width) / 2.0,
        y=pad,
        scale=structure_scale,
        effective_font_size=structure_effective_font_size,
    )
    component_source_title_omitted_count = _append_panel(
        root,
        asset=component,
        panel="component_span",
        row=2,
        x=pad + (content_width - component_width) / 2.0,
        y=pad + structure_height + gap,
        scale=component_scale,
        effective_font_size=component_effective_font_size,
    )
    return _ReviewSvgBuild(
        root=root,
        component_source_title_omitted_count=component_source_title_omitted_count,
    )


def _append_panel(
    root: ET.Element,
    *,
    asset: _SvgAsset,
    panel: str,
    row: int,
    x: float,
    y: float,
    scale: float,
    effective_font_size: float,
) -> int:
    nested = ET.SubElement(
        root,
        f"{{{SVG_NS}}}svg",
        {
            "x": f"{x:.3f}",
            "y": f"{y:.3f}",
            "width": f"{asset.width * scale:.3f}",
            "height": f"{asset.height * scale:.3f}",
            "viewBox": " ".join(f"{part:.3f}" for part in asset.viewbox),
            "data-dnadesign-panel": panel,
            "data-dnadesign-panel-row": str(row),
            "data-dnadesign-source-svg": asset.path.name,
            "data-dnadesign-source-nucleotide-font-size-px": f"{asset.source_nucleotide_font_size_px:.3f}",
            "data-dnadesign-effective-nucleotide-font-size-px": f"{effective_font_size:.3f}",
            "overflow": "visible",
        },
    )
    omitted_count = 0
    if panel == "component_span":
        nested.set("data-dnadesign-component-panel-emphasis", _COMPONENT_PANEL_EMPHASIS)
        nested.set("data-dnadesign-component-source-title-policy", _COMPONENT_SOURCE_TITLE_POLICY)
    for child in list(asset.root):
        child_copy = copy.deepcopy(child)
        if panel == "component_span":
            omitted_count += _remove_component_source_title_groups(child_copy, source_height=asset.height)
            _apply_component_panel_emphasis(child_copy)
        nested.append(child_copy)
    if panel == "component_span":
        nested.set("data-dnadesign-component-source-title-omitted-count", str(omitted_count))
    return omitted_count


def _manifest(
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
    composition_id = _composition_id_from_visual_contract(visual_contract)
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
        artifacts={"review_svg": COMPOSITION_REVIEW_SVG_PATH.as_posix()},
        layout={
            "row_count": 2,
            "panel_order": ["secondary_structure", "component_span"],
            "component_nucleotide_font_size_px": target_nucleotide_font_size_px,
            "structure_fit_policy": _STRUCTURE_FIT_POLICY,
            "structure_scale": structure_scale,
            "component_scale": component_scale,
            "structure_width_px": structure_width_px,
            "component_width_px": component_width_px,
            "structure_effective_nucleotide_font_size_px": structure_effective_font_size,
            "component_effective_nucleotide_font_size_px": component_effective_font_size,
            "component_panel_emphasis": _COMPONENT_PANEL_EMPHASIS,
            "component_source_title_policy": _COMPONENT_SOURCE_TITLE_POLICY,
            "structure_to_component_width_ratio": width_ratio,
            "vertical_gap_px": 18.0,
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


def _remove_component_source_title_groups(element: ET.Element, *, source_height: float) -> int:
    removed_count = 0
    for child in list(element):
        if _is_component_source_title_group(child, source_height=source_height):
            element.remove(child)
            removed_count += 1
            continue
        removed_count += _remove_component_source_title_groups(child, source_height=source_height)
    return removed_count


def _is_component_source_title_group(element: ET.Element, *, source_height: float) -> bool:
    if _local_name(element.tag) != "g":
        return False
    if not str(element.attrib.get("id", "")).startswith("text_"):
        return False
    if not _contains_fill_color(element, _COMPONENT_SOURCE_TITLE_COLOR):
        return False
    y_values = [
        y
        for node in element.iter()
        if (y := _translate_y_from_transform(str(node.attrib.get("transform", "")))) is not None
    ]
    if not y_values:
        return False
    return min(y_values) <= max(12.0, source_height * 0.24)


def _contains_fill_color(element: ET.Element, color: str) -> bool:
    normalized_color = _normalize_css_text(color)
    return any(normalized_color in _normalize_css_text(str(node.attrib.get("style", ""))) for node in element.iter())


def _translate_y_from_transform(transform: str) -> float | None:
    match = re.search(r"translate\(([^)]*)\)", transform)
    if match is None:
        return None
    values = _parse_float_list(match.group(1))
    if len(values) < 2:
        return 0.0 if values else None
    return values[1]


def _normalize_css_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text).strip().lower())


def _apply_component_panel_emphasis(element: ET.Element) -> None:
    for node in element.iter():
        if _local_name(node.tag) != "use":
            continue
        node.set("data-dnadesign-review-emphasis", "component_span_bold_glyph")
        style = str(node.attrib.get("style", "")).strip()
        node.set(
            "style",
            _append_style_declarations(
                style,
                (
                    "stroke: #0F172A",
                    "stroke-width: 0.28px",
                    "stroke-opacity: 0.62",
                    "paint-order: stroke fill",
                ),
            ),
        )


def _composition_id_from_visual_contract(visual_contract: SequenceEvidenceMapV1) -> str:
    source_id = visual_contract.meta.get("visual_scope", {}).get("source_sequence_id")
    if isinstance(source_id, str) and source_id.strip():
        return source_id.strip()
    return visual_contract.state_id.rsplit(".", maxsplit=1)[0]


def _viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    raw = str(root.attrib.get("viewBox", "")).strip()
    if raw:
        values = _parse_float_list(raw)
        if len(values) == 4:
            return values[0], values[1], values[2], values[3]
    width = _numeric_length(root.attrib.get("width"))
    height = _numeric_length(root.attrib.get("height"))
    if width is not None and height is not None:
        return 0.0, 0.0, width, height
    raise ValidationError("SVG must provide either viewBox or width/height.")


def _extract_nucleotide_font_size_px(root: ET.Element, *, default: float) -> float:
    for node in root.iter():
        if _local_name(node.tag) != "text":
            continue
        classes = set(str(node.attrib.get("class", "")).split())
        if "nucleotide" not in classes:
            continue
        font_size = _font_size_from_style(str(node.attrib.get("style", "")))
        if font_size is not None:
            return font_size
        font_size = _numeric_length(node.attrib.get("font-size"))
        if font_size is not None:
            return font_size
    style_text = "\n".join(str(node.text or "") for node in root.iter() if _local_name(node.tag) == "style")
    match = re.search(r"\.nucleotide\s*\{[^}]*font-size\s*:\s*([0-9.]+)", style_text, flags=re.IGNORECASE)
    if match is not None:
        return float(match.group(1))
    return default


def _font_size_from_style(style: str) -> float | None:
    match = re.search(r"(?:^|;)\s*font-size\s*:\s*([0-9.]+)", style, flags=re.IGNORECASE)
    if match is None:
        return None
    return float(match.group(1))


def _append_style_declarations(style: str, declarations: tuple[str, ...]) -> str:
    existing = style.strip()
    suffix = "; ".join(item.strip().rstrip(";") for item in declarations if item.strip())
    if not suffix:
        return existing
    if existing and not existing.endswith(";"):
        existing = f"{existing};"
    return f"{existing} {suffix};".strip() if existing else f"{suffix};"


def _numeric_length(value: object) -> float | None:
    if value is None:
        return None
    match = re.match(r"\s*([-+]?\d+(?:\.\d+)?)", str(value))
    if match is None:
        return None
    return float(match.group(1))


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in re.split(r"[\s,]+", raw.strip()):
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", maxsplit=1)[-1]
    return tag


def _update_bundle_manifest(bundle: Path, review_manifest: CompositionReviewSvgV1) -> None:
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
    _write_json(manifest_path, payload)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "COMPONENT_SPAN_SVG_PATH",
    "COMPOSITION_REVIEW_MANIFEST_PATH",
    "COMPOSITION_REVIEW_SVG_PATH",
    "STRUCTURE_SVG_PATH",
    "publish_composition_review_svg",
]
