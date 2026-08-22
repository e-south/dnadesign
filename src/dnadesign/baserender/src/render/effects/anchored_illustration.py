"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/effects/anchored_illustration.py

Anchored, collision-aware illustration effects for sequence rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MatplotlibPath

from ...config import Style
from ...core import Effect, Record, RenderingError
from ..layout import LayoutContext
from ..palette import Palette

_ASSET_SCHEMA = "baserender.anchored_illustration_asset.v1"
_ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets" / "overlays"


@dataclass(frozen=True)
class AnchoredIllustrationAsset:
    asset_id: str
    image: object
    aspect_ratio: float
    anchors: Mapping[str, tuple[float, float]]


@dataclass(frozen=True)
class AnchoredIllustrationBinding:
    anchor_id: str
    feature_id: str
    start: int
    end: int


@dataclass(frozen=True)
class AnchoredIllustrationGeometry:
    x0: float
    y0: float
    x1: float
    y1: float
    anchors: Mapping[str, tuple[float, float]]
    bindings: tuple[AnchoredIllustrationBinding, ...]

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


def _required_text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RenderingError(f"{field_name} must be a non-empty string")
    return text


def _finite_positive(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderingError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RenderingError(f"{field_name} must be finite and > 0")
    return number


@lru_cache(maxsize=8)
def _load_asset(asset_id: str) -> AnchoredIllustrationAsset:
    if Path(asset_id).name != asset_id:
        raise RenderingError("anchored illustration asset ids must not contain paths")
    manifest_path = _ASSET_ROOT / f"{asset_id}.json"
    if not manifest_path.is_file():
        raise RenderingError(f"Unknown anchored illustration asset: {asset_id!r}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RenderingError(f"Could not load anchored illustration manifest: {manifest_path}") from exc
    if payload.get("schema") != _ASSET_SCHEMA:
        raise RenderingError(f"Unsupported anchored illustration asset schema: {payload.get('schema')!r}")
    if payload.get("asset_id") != asset_id:
        raise RenderingError("anchored illustration manifest asset_id mismatch")
    image_name = _required_text(payload.get("image"), field_name="asset.image")
    if Path(image_name).name != image_name:
        raise RenderingError("asset.image must be a file name")
    pixel_size = payload.get("pixel_size")
    if not isinstance(pixel_size, Sequence) or isinstance(pixel_size, (str, bytes)) or len(pixel_size) != 2:
        raise RenderingError("asset.pixel_size must contain width and height")
    width = _finite_positive(pixel_size[0], field_name="asset.pixel_size[0]")
    height = _finite_positive(pixel_size[1], field_name="asset.pixel_size[1]")
    raw_anchors = payload.get("anchors")
    if not isinstance(raw_anchors, Mapping) or not raw_anchors:
        raise RenderingError("asset.anchors must be a non-empty mapping")
    anchors: dict[str, tuple[float, float]] = {}
    for key, raw in raw_anchors.items():
        if not isinstance(raw, Mapping):
            raise RenderingError(f"asset.anchors.{key} must be a mapping")
        x = raw.get("x_fraction")
        y = raw.get("y_fraction_from_top")
        if (
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or isinstance(y, bool)
            or not isinstance(y, (int, float))
        ):
            raise RenderingError(f"asset.anchors.{key} fractions must be numeric")
        point = (float(x), float(y))
        if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in point):
            raise RenderingError(f"asset.anchors.{key} fractions must be within [0, 1]")
        anchors[_required_text(key, field_name="asset anchor id")] = point
    image_path = _ASSET_ROOT / image_name
    if not image_path.is_file():
        raise RenderingError(f"anchored illustration image is missing: {image_path}")
    try:
        image = mpimg.imread(image_path)
    except (OSError, ValueError) as exc:
        raise RenderingError(f"Could not load anchored illustration image: {image_path}") from exc
    return AnchoredIllustrationAsset(
        asset_id=asset_id,
        image=image,
        aspect_ratio=height / width,
        anchors=anchors,
    )


def _bindings(
    effect: Effect, record: Record, asset: AnchoredIllustrationAsset
) -> tuple[AnchoredIllustrationBinding, ...]:
    raw_bindings = effect.target.get("bindings")
    if not isinstance(raw_bindings, Sequence) or isinstance(raw_bindings, (str, bytes)) or not raw_bindings:
        raise RenderingError("anchored_illustration target.bindings must be a non-empty list")
    bindings: list[AnchoredIllustrationBinding] = []
    seen_anchors: set[str] = set()
    seen_features: set[str] = set()
    for index, raw in enumerate(raw_bindings):
        if not isinstance(raw, Mapping):
            raise RenderingError(f"anchored_illustration target.bindings[{index}] must be a mapping")
        anchor_id = _required_text(raw.get("anchor_id"), field_name=f"target.bindings[{index}].anchor_id")
        feature_id = _required_text(raw.get("feature_id"), field_name=f"target.bindings[{index}].feature_id")
        if anchor_id not in asset.anchors:
            raise RenderingError(f"anchored illustration asset {asset.asset_id!r} has no anchor {anchor_id!r}")
        if anchor_id in seen_anchors or feature_id in seen_features:
            raise RenderingError("anchored illustration bindings must use distinct anchors and features")
        start = raw.get("start")
        end = raw.get("end")
        if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int):
            raise RenderingError(f"target.bindings[{index}] start/end must be integers")
        if start < 0 or end <= start or end > len(record.sequence):
            raise RenderingError(f"target.bindings[{index}] span is outside the record sequence")
        seen_anchors.add(anchor_id)
        seen_features.add(feature_id)
        bindings.append(AnchoredIllustrationBinding(anchor_id, feature_id, start, end))
    return tuple(bindings)


def compute_anchored_illustration_geometry(
    effect: Effect,
    record: Record,
    layout: LayoutContext,
    style: Style,
) -> tuple[AnchoredIllustrationAsset, AnchoredIllustrationGeometry]:
    asset_id = _required_text(effect.params.get("asset_id"), field_name="anchored_illustration params.asset_id")
    asset = _load_asset(asset_id)
    bindings = _bindings(effect, record, asset)
    width = _finite_positive(effect.params.get("width_px", 300.0), field_name="anchored_illustration params.width_px")
    height = width * asset.aspect_ratio
    planned_centers = [layout.x_left + ((item.start + item.end) / 2.0) * layout.cw for item in bindings]
    anchor_center = sum(asset.anchors[item.anchor_id][0] for item in bindings) / len(bindings)
    x0 = (sum(planned_centers) / len(planned_centers)) - anchor_center * width
    minimum_x = float(style.padding_x)
    maximum_x = float(layout.width - style.padding_x - width)
    if maximum_x < minimum_x:
        raise RenderingError("anchored illustration is wider than the available sequence-row canvas")
    x0 = min(maximum_x, max(minimum_x, x0))
    x1 = x0 + width
    top_gap = _finite_positive(
        effect.params.get("top_gap_px", 8.0), field_name="anchored_illustration params.top_gap_px"
    )
    y1 = float(layout.content_top) - top_gap
    y0 = y1 - height
    if y0 <= float(layout.content_bottom):
        raise RenderingError("anchored illustration has no reserved vertical lane")
    anchor_points = {key: (x0 + point[0] * width, y1 - point[1] * height) for key, point in asset.anchors.items()}
    return asset, AnchoredIllustrationGeometry(x0, y0, x1, y1, anchor_points, bindings)


def anchored_illustration_occupied_boxes(
    effect: Effect,
    record: Record,
    layout: LayoutContext,
    style: Style,
    feature_boxes: Mapping[str, tuple[float, float, float, float]],
) -> tuple[tuple[float, float, float, float], ...]:
    _asset, geometry = compute_anchored_illustration_geometry(effect, record, layout, style)
    padding = 6.0
    del feature_boxes
    return ((geometry.x0 - padding, geometry.y0 - padding, geometry.x1 + padding, geometry.y1 + padding),)


def validate_anchored_illustration(
    effect: Effect,
    record: Record,
    layout: LayoutContext,
    style: Style,
    palette: Palette,
    feature_boxes: dict[str, tuple[float, float, float, float]],
) -> None:
    del palette, feature_boxes
    compute_anchored_illustration_geometry(effect, record, layout, style)
    color = str(effect.params.get("fill_color", "#DDE2E7"))
    if not mcolors.is_color_like(color):
        raise RenderingError("anchored_illustration params.fill_color must be a valid color")
    alpha = effect.params.get("fill_alpha", 0.42)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise RenderingError("anchored_illustration params.fill_alpha must be numeric")
    if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) <= 1.0:
        raise RenderingError("anchored_illustration params.fill_alpha must be within (0, 1]")


def draw_anchored_illustration(
    ax,
    effect: Effect,
    record: Record,
    layout: LayoutContext,
    style: Style,
    palette: Palette,
    feature_boxes: dict[str, tuple[float, float, float, float]],
) -> None:
    del palette
    asset, geometry = compute_anchored_illustration_geometry(effect, record, layout, style)
    color = str(effect.params.get("fill_color", "#DDE2E7"))
    alpha = float(effect.params.get("fill_alpha", 0.42))
    visible = []
    for binding in geometry.bindings:
        feature_box = feature_boxes.get(binding.feature_id)
        if feature_box is None:
            continue
        visible.append((geometry.anchors[binding.anchor_id], feature_box))
    if visible:
        for anchor, feature_box in visible:
            shoulder = max(8.0, (geometry.x1 - geometry.x0) * 0.025)
            vertices = (
                (anchor[0] - shoulder, anchor[1]),
                (anchor[0] + shoulder, anchor[1]),
                (feature_box[2], feature_box[3] + 1.0),
                (feature_box[0], feature_box[3] + 1.0),
                (anchor[0] - shoulder, anchor[1]),
            )
            path = MatplotlibPath(
                vertices,
                (
                    MatplotlibPath.MOVETO,
                    MatplotlibPath.LINETO,
                    MatplotlibPath.LINETO,
                    MatplotlibPath.LINETO,
                    MatplotlibPath.CLOSEPOLY,
                ),
            )
            patch = PathPatch(
                path,
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
                alpha=alpha,
                zorder=1.35,
            )
            patch.set_gid(f"anchored_illustration_footprint:{asset.asset_id}:{anchor[0]:.3f}")
            ax.add_artist(patch)
    image = ax.imshow(
        asset.image,
        extent=(geometry.x0, geometry.x1, geometry.y0, geometry.y1),
        origin="upper",
        interpolation="lanczos",
        aspect="auto",
        zorder=2.7,
    )
    image.set_gid(f"anchored_illustration:{asset.asset_id}")
