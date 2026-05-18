"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/retron_hairpin_design/outputs/materialized_outputs.py

Retron MSD materialized artifact publication and SVG rasterization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

from ..errors import RetronMsdCompilerError
from .layout import (
    BASERENDER_CONTRACT_KIND,
    VARIANT_MANIFEST_COMPOSITION_DIRNAME,
    VARIANT_MANIFEST_CONSTRUCT_DIRNAME,
    VARIANT_MANIFEST_DIRNAME,
    VARIANT_MANIFEST_FOLDING_DIRNAME,
    VARIANT_MANIFEST_PROVENANCE_DIRNAME,
    VARIANT_MANIFEST_REVIEWS_DIRNAME,
    VARIANT_MANIFEST_SECONDARY_STRUCTURE_DIRNAME,
    VARIANT_MANIFEST_VISUAL_DIRNAME,
    VARIANT_PLOTS_DIRNAME,
    VARIANT_SEQUENCES_DIRNAME,
)


def run_baserender_jobs(
    artifact_bundle: Path,
    *,
    formats: Sequence[str],
    enabled: bool,
) -> dict[str, str]:
    rendered: dict[str, str] = {}
    if not enabled:
        return rendered
    import dnadesign.baserender as baserender

    for fmt in formats:
        job_path = artifact_bundle / "baserender_jobs" / f"component_span_qa_{fmt}.yaml"
        try:
            report = baserender.run_job(
                job_path,
                kind=BASERENDER_CONTRACT_KIND,
                strict=True,
                caller_root=artifact_bundle,
            )
        except Exception as exc:  # pragma: no cover - depends on renderer backend failure mode.
            raise RetronMsdCompilerError(f"BaseRender failed for Retron MSD bundle '{artifact_bundle}': {exc}") from exc
        image_path = Path(report.outputs["images_path"])
        flat_path = artifact_bundle / f"component_span_qa.{fmt}"
        if image_path != flat_path:
            shutil.copyfile(image_path, flat_path)
        rendered[f"component_span_{fmt}"] = flat_path.as_posix()
    return rendered


def publish_variant_outputs(
    variant_dir: Path,
    *,
    construct_bundle: Path,
    root: Path,
) -> dict[str, object]:
    sequences_dir = variant_dir / VARIANT_SEQUENCES_DIRNAME
    plots_dir = variant_dir / VARIANT_PLOTS_DIRNAME
    manifest_dir = variant_dir / VARIANT_MANIFEST_DIRNAME
    sequences_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    forward_genbank = sequences_dir / "forward.gb"
    reverse_complement_genbank = sequences_dir / "reverse_complement.gb"
    forward_fasta = sequences_dir / "forward.fa"
    reverse_complement_fasta = sequences_dir / "reverse_complement.fa"
    features_csv = sequences_dir / "features.csv"
    _copy_required_file(construct_bundle / "sequence.gb", forward_genbank)
    _copy_required_file(construct_bundle / "sequence.reverse_complement.gb", reverse_complement_genbank)
    _copy_required_file(construct_bundle / "sequence.fa", forward_fasta)
    _copy_required_file(construct_bundle / "sequence.reverse_complement.fa", reverse_complement_fasta)
    _copy_required_file(construct_bundle / "features.csv", features_csv)

    folding_prediction = construct_bundle / "folding" / "secondary_structure_prediction_v1.json"
    folding_status = _folding_prediction_status(folding_prediction)
    structure_plot_dir = construct_bundle / "visual" / "viennarna_secondary_structure"
    structure_manifest = structure_plot_dir / "viennarna_secondary_structure_svg_v1.json"
    structure_annotation_manifest = structure_plot_dir / "secondary_structure.annotation_manifest.json"
    native_structure_svg = structure_plot_dir / "secondary_structure.native.svg"
    native_structure_png = plots_dir / "secondary_structure.native.png"
    _rasterize_svg_to_png(native_structure_svg, native_structure_png)

    review_manifest = _publish_composition_review(construct_bundle)
    composition_overview_svg = plots_dir / "composition_overview.svg"
    composition_overview_png = plots_dir / "composition_overview.png"
    _copy_required_file(construct_bundle / review_manifest.artifacts.review_svg, composition_overview_svg)
    _copy_required_file(construct_bundle / review_manifest.artifacts.review_png, composition_overview_png)

    composition_manifest_dir = manifest_dir / VARIANT_MANIFEST_COMPOSITION_DIRNAME
    construct_manifest_dir = manifest_dir / VARIANT_MANIFEST_CONSTRUCT_DIRNAME
    folding_manifest_dir = manifest_dir / VARIANT_MANIFEST_FOLDING_DIRNAME
    provenance_manifest_dir = manifest_dir / VARIANT_MANIFEST_PROVENANCE_DIRNAME
    reviews_manifest_dir = manifest_dir / VARIANT_MANIFEST_REVIEWS_DIRNAME
    visual_manifest_dir = manifest_dir / VARIANT_MANIFEST_VISUAL_DIRNAME
    secondary_structure_manifest_dir = visual_manifest_dir / VARIANT_MANIFEST_SECONDARY_STRUCTURE_DIRNAME
    manifest_sources = [
        (construct_bundle / "manifest.json", construct_manifest_dir / "manifest.json"),
        (construct_bundle / "assembled_sequence.json", composition_manifest_dir / "assembled_sequence.json"),
        (construct_bundle / "segment_spans.json", composition_manifest_dir / "segment_spans.json"),
        (construct_bundle / "annotation_spans.json", composition_manifest_dir / "annotation_spans.json"),
        (construct_bundle / "provenance.json", provenance_manifest_dir / "provenance.json"),
        (construct_bundle / "validation_report.json", composition_manifest_dir / "validation_report.json"),
        (
            construct_bundle / "visual" / "sequence_evidence_map_v1.json",
            visual_manifest_dir / "sequence_evidence_map_v1.json",
        ),
        (folding_prediction, folding_manifest_dir / "secondary_structure_prediction_v1.json"),
        (construct_bundle / "folding" / "folding_preflight.json", folding_manifest_dir / "folding_preflight.json"),
        (
            construct_bundle / "folding" / "secondary_structure_prediction_request_v1.yaml",
            folding_manifest_dir / "secondary_structure_prediction_request_v1.yaml",
        ),
        (structure_manifest, secondary_structure_manifest_dir / "viennarna_secondary_structure_svg_v1.json"),
        (structure_annotation_manifest, secondary_structure_manifest_dir / "annotation_manifest.json"),
        (native_structure_svg, secondary_structure_manifest_dir / "native.svg"),
        (
            structure_plot_dir / "secondary_structure.annotated.svg",
            secondary_structure_manifest_dir / "annotated.svg",
        ),
        (
            construct_bundle / "visual" / "reviews" / "composition_review_svg_v1.json",
            reviews_manifest_dir / "composition_review_svg_v1.json",
        ),
    ]
    for source, destination in manifest_sources:
        _copy_required_file(source, destination)

    return {
        "genbank": forward_genbank.relative_to(root).as_posix(),
        "reverse_complement_genbank": reverse_complement_genbank.relative_to(root).as_posix(),
        "forward_fasta": forward_fasta.relative_to(root).as_posix(),
        "reverse_complement_fasta": reverse_complement_fasta.relative_to(root).as_posix(),
        "features_csv": features_csv.relative_to(root).as_posix(),
        "visual_contract": (visual_manifest_dir / "sequence_evidence_map_v1.json").relative_to(root).as_posix(),
        "construct_manifest": (construct_manifest_dir / "manifest.json").relative_to(root).as_posix(),
        "folding_prediction": (folding_manifest_dir / "secondary_structure_prediction_v1.json")
        .relative_to(root)
        .as_posix(),
        "folding_status": folding_status,
        "composition_overview_svg": composition_overview_svg.relative_to(root).as_posix(),
        "composition_overview_png": composition_overview_png.relative_to(root).as_posix(),
        "secondary_structure_native_png": native_structure_png.relative_to(root).as_posix(),
    }


def _publish_composition_review(construct_bundle: Path):
    import dnadesign.construct as construct

    try:
        return construct.publish_composition_review_svg(construct_bundle)
    except Exception as exc:  # pragma: no cover - depends on producer validation failure mode.
        raise RetronMsdCompilerError(
            f"Construct failed to publish Retron MSD composition review for '{construct_bundle}': {exc}"
        ) from exc


def _rasterize_svg_to_png(source_svg: Path, output_png: Path) -> None:
    if not source_svg.is_file():
        raise RetronMsdCompilerError(f"Expected ViennaRNA native SVG artifact is missing: {source_svg}")
    try:
        _write_viennarna_native_svg_png(source_svg, output_png)
    except Exception as exc:
        if isinstance(exc, RetronMsdCompilerError):
            raise
        raise RetronMsdCompilerError(
            f"Failed to rasterize ViennaRNA native SVG '{source_svg}' to PNG with the built-in renderer: {exc}"
        ) from exc


def _write_viennarna_native_svg_png(source_svg: Path, output_png: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - dependency is pinned in the managed environment.
        raise RetronMsdCompilerError("Pillow is required to publish ViennaRNA native PNG artifacts.") from exc

    tree = ET.parse(source_svg)
    root = tree.getroot()
    width, height = _svg_dimensions(root)
    raster_scale = 2
    image = Image.new("RGB", (width * raster_scale, height * raster_scale), color="white")
    draw = ImageDraw.Draw(image)
    font_size = 12 * raster_scale
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # pragma: no cover - old Pillow compatibility.
        font = ImageFont.load_default()

    def render(element: ET.Element, transforms: list[tuple[str, tuple[float, ...]]]) -> None:
        tag = _svg_tag(element)
        next_transforms = [*transforms, *_parse_svg_transform(element.get("transform", ""))]
        if tag == "polyline":
            points = [
                _scale_point(_apply_svg_transform(point, next_transforms), raster_scale)
                for point in _svg_points(element)
            ]
            if len(points) >= 2:
                draw.line(points, fill=_svg_stroke(element), width=_scaled_stroke_width(element, raster_scale))
        elif tag == "line":
            start = _apply_svg_transform(
                (_float_attr(element, "x1"), _float_attr(element, "y1")),
                next_transforms,
            )
            end = _apply_svg_transform(
                (_float_attr(element, "x2"), _float_attr(element, "y2")),
                next_transforms,
            )
            draw.line(
                [_scale_point(start, raster_scale), _scale_point(end, raster_scale)],
                fill=_svg_stroke(element),
                width=_scaled_stroke_width(element, raster_scale),
            )
        elif tag == "text":
            point = _apply_svg_transform(
                (_float_attr(element, "x"), _float_attr(element, "y")),
                next_transforms,
            )
            x, y = _scale_point(point, raster_scale)
            draw.text((x, y - font_size), element.text or "", fill="#000000", font=font)
        elif tag in {"rect", "script", "style"}:
            pass

        for child in element:
            render(child, next_transforms)

    render(root, [])
    if raster_scale != 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)


def _svg_dimensions(root: ET.Element) -> tuple[int, int]:
    width = _optional_svg_float(root.get("width"))
    height = _optional_svg_float(root.get("height"))
    if width is not None and height is not None:
        return max(1, round(width)), max(1, round(height))
    view_box = root.get("viewBox")
    if view_box:
        parts = [float(part) for part in view_box.replace(",", " ").split()]
        if len(parts) == 4:
            return max(1, round(parts[2])), max(1, round(parts[3]))
    raise RetronMsdCompilerError("ViennaRNA native SVG is missing width/height or viewBox dimensions.")


def _svg_tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", maxsplit=1)[-1]


def _parse_svg_transform(value: str) -> list[tuple[str, tuple[float, ...]]]:
    transforms: list[tuple[str, tuple[float, ...]]] = []
    for chunk in value.replace(",", " ").split(")"):
        chunk = chunk.strip()
        if not chunk or "(" not in chunk:
            continue
        name, raw_args = chunk.split("(", 1)
        args = tuple(float(part) for part in raw_args.split())
        transform_name = name.strip()
        if transform_name in {"translate", "scale"}:
            transforms.append((transform_name, args))
    return transforms


def _apply_svg_transform(
    point: tuple[float, float],
    transforms: Sequence[tuple[str, tuple[float, ...]]],
) -> tuple[float, float]:
    x, y = point
    for name, args in reversed(transforms):
        if name == "translate":
            x += args[0] if args else 0.0
            y += args[1] if len(args) > 1 else 0.0
        elif name == "scale":
            sx = args[0] if args else 1.0
            sy = args[1] if len(args) > 1 else sx
            x *= sx
            y *= sy
    return x, y


def _scale_point(point: tuple[float, float], scale: int) -> tuple[float, float]:
    return point[0] * scale, point[1] * scale


def _svg_points(element: ET.Element) -> list[tuple[float, float]]:
    raw_points = str(element.get("points") or "").replace(",", " ").split()
    numbers = [float(value) for value in raw_points]
    return list(zip(numbers[0::2], numbers[1::2], strict=False))


def _svg_stroke(element: ET.Element) -> str:
    class_name = str(element.get("class") or "")
    if class_name == "basepairs":
        return "#ff0000"
    if class_name == "backbone":
        return "#808080"
    style = _style_map(element)
    return style.get("stroke") or "#000000"


def _scaled_stroke_width(element: ET.Element, raster_scale: int) -> int:
    class_name = str(element.get("class") or "")
    if class_name == "basepairs":
        width = 2.5
    elif class_name == "backbone":
        width = 1.5
    else:
        width = _optional_svg_float(_style_map(element).get("stroke-width")) or 1.0
    return max(1, round(width * raster_scale))


def _style_map(element: ET.Element) -> dict[str, str]:
    style: dict[str, str] = {}
    for part in str(element.get("style") or "").split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        style[key.strip()] = value.strip()
    return style


def _float_attr(element: ET.Element, name: str) -> float:
    value = _optional_svg_float(element.get(name))
    if value is None:
        raise RetronMsdCompilerError(f"ViennaRNA native SVG element is missing numeric '{name}' attribute.")
    return value


def _optional_svg_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    for suffix in ("px", "pt"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    try:
        return float(text)
    except ValueError:
        return None


def _copy_required_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RetronMsdCompilerError(f"Expected MSD materialize artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _folding_prediction_status(path: Path) -> str:
    if not path.is_file():
        raise RetronMsdCompilerError(
            f"Expected folding prediction artifact is missing: {path}. "
            "Materialize enables folding by default and requires an explicit folding status artifact."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = str(payload.get("status") or "").strip()
    if not status:
        raise RetronMsdCompilerError(f"Folding prediction artifact is missing status: {path}")
    if status != "ok":
        raise RetronMsdCompilerError(
            f"Retron MSD materialize requires a ViennaRNA-backed folding prediction with status ok: "
            f"{path} reported {status}."
        )
    return status


__all__ = ["publish_variant_outputs", "run_baserender_jobs"]
