"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/three_way_junction_review.py

Four-panel QA renderer for neutral three-way-junction review evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from pydantic import ValidationError

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from ..config import Style
from ..core import Record, RenderingError, SchemaError
from ..core.pydantic_validation import format_validation_error
from .palette import Palette
from .sequence_preview import bounded_sequence_preview, bounded_text_preview

_INK = "#172033"
_MUTED = "#667085"
_GRID = "#D7DCE4"
_FRAGMENT = "#CBD5E1"
_TOEHOLD = "#5B8DEF"
_BARCODE = "#2A9D8F"
_RECOVERY = "#D97706"
_WARNING = "#B45309"
_WARNING_BG = "#FFF3D6"

_AXIS_GIDS = (
    "three-way-junction-review:target-geometry",
    "three-way-junction-review:junction-assignments",
    "three-way-junction-review:strands-and-recovery",
    "three-way-junction-review:search-and-checks",
)

_MAX_DISPLAYED_LOCI = 6
_FIGURE_WIDTH_INCHES = 15.2
_FIGURE_HEIGHT_INCHES = 4.2
_MAX_REVIEW_DPI = 600
_MAX_REVIEW_FIGURE_SCALE = 4.0
_MAX_REVIEW_CANVAS_DIMENSION_PX = 16_384
_MAX_REVIEW_CANVAS_RGBA_BYTES = 64 * 1024 * 1024
_RGBA_BYTES_PER_PIXEL = 4


def _review_figure_size(style: Style) -> tuple[float, float]:
    """Validate and return the renderer-owned figure size in inches."""

    try:
        dpi = float(style.dpi)
        figure_scale = float(style.figure_scale)
    except (TypeError, ValueError, OverflowError):
        raise SchemaError("three_way_junction_review style dimensions must be finite numbers") from None
    if not math.isfinite(dpi):
        raise SchemaError("three_way_junction_review style.dpi must be finite")
    if not math.isfinite(figure_scale):
        raise SchemaError("three_way_junction_review style.figure_scale must be finite")
    if dpi > _MAX_REVIEW_DPI:
        raise SchemaError("three_way_junction_review style.dpi exceeds the renderer limit")
    if figure_scale > _MAX_REVIEW_FIGURE_SCALE:
        raise SchemaError("three_way_junction_review style.figure_scale exceeds the renderer limit")

    figure_width = _FIGURE_WIDTH_INCHES * figure_scale
    figure_height = _FIGURE_HEIGHT_INCHES * figure_scale
    width_px = math.ceil(figure_width * dpi)
    height_px = math.ceil(figure_height * dpi)
    if max(width_px, height_px) > _MAX_REVIEW_CANVAS_DIMENSION_PX:
        raise SchemaError("three_way_junction_review canvas dimension exceeds the renderer limit")
    if width_px * height_px * _RGBA_BYTES_PER_PIXEL > _MAX_REVIEW_CANVAS_RGBA_BYTES:
        raise SchemaError("three_way_junction_review canvas exceeds the 64 MiB RGBA allocation limit")
    return figure_width, figure_height


def _display_indices(length: int) -> tuple[int, ...]:
    if length <= _MAX_DISPLAYED_LOCI:
        return tuple(range(length))
    side = _MAX_DISPLAYED_LOCI // 2
    return (*range(side), *range(length - side, length))


def _integer_preview(value: int) -> tuple[str, bool]:
    if abs(value) < 10**18:
        return str(value), False
    magnitude = abs(value)
    digit_count = max(1, (magnitude.bit_length() * 30_103) // 100_000 + 1)
    while magnitude < 10 ** (digit_count - 1):
        digit_count -= 1
    while magnitude >= 10**digit_count:
        digit_count += 1
    leading = magnitude // (10 ** (digit_count - 4))
    trailing = magnitude % 10**4
    byte_length = max(1, (magnitude.bit_length() + 7) // 8)
    digest = hashlib.sha256(magnitude.to_bytes(byte_length, "big")).hexdigest()[:12]
    sign = "-" if value < 0 else ""
    return f"{digit_count} digits · {digest} · {sign}{leading:04d}…{trailing:04d}", True


def _review_from_record(record: Record) -> ThreeWayJunctionReviewV1:
    meta = record.meta if isinstance(record.meta, Mapping) else None
    if meta is None:
        raise RenderingError("three_way_junction_review requires record.meta.three_way_junction_review")
    payload = meta.get("three_way_junction_review")
    if not isinstance(payload, Mapping):
        raise RenderingError("three_way_junction_review requires record.meta.three_way_junction_review")
    try:
        return ThreeWayJunctionReviewV1.model_validate(payload)
    except ValidationError as exc:
        detail = format_validation_error(exc)
        raise RenderingError(f"three_way_junction_review received invalid review evidence: {detail}") from None


def _setup_axis(axis, *, title: str, gid: str) -> None:
    axis.set_gid(gid)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(title, loc="left", fontsize=11, fontweight="semibold", color=_INK, pad=10)


def _panel_geometry(axis, review: ThreeWayJunctionReviewV1) -> None:
    target_length = len(review.target.sequence_5to3)
    target_id = bounded_text_preview(review.target.target_id)
    if target_id.abbreviated:
        axis.text(
            0.02,
            0.92,
            f"Target ID · {target_id.length_chars} chars · SHA-256[:12] {target_id.sha256_prefix}",
            fontsize=6.5,
            fontweight="semibold",
            color=_INK,
            va="top",
        )
        axis.text(0.02, 0.86, f"preview {target_id.preview}", fontsize=6.5, family="monospace", color=_INK, va="top")
        summary_y = 0.78
    else:
        axis.text(0.02, 0.92, target_id.preview, fontsize=10, fontweight="semibold", color=_INK, va="top")
        summary_y = 0.83
    axis.text(
        0.02,
        summary_y,
        f"{target_length} nt  ·  {len(review.geometry.fragments)} fragments  ·  {len(review.geometry.junctions)} "
        f"junction{'s' if len(review.geometry.junctions) != 1 else ''}",
        fontsize=8.2,
        color=_MUTED,
        va="top",
    )
    y = 0.55
    axis.plot([0.05, 0.95], [y, y], color=_INK, linewidth=1.2, zorder=1)
    fragment_indices = set(_display_indices(len(review.geometry.fragments)))
    junction_indices = set(_display_indices(len(review.geometry.junctions)))
    for index, fragment in enumerate(review.geometry.fragments):
        if index not in fragment_indices:
            continue
        x = 0.05 + 0.90 * fragment.domain_span.start / target_length
        width = 0.90 * (fragment.domain_span.end - fragment.domain_span.start) / target_length
        axis.add_patch(Rectangle((x, y - 0.08), width, 0.16, facecolor=_FRAGMENT, edgecolor="white", linewidth=1.0))
        axis.text(x + width / 2, y, f"F{fragment.index + 1}", ha="center", va="center", fontsize=8, color=_INK)
    for index, junction in enumerate(review.geometry.junctions):
        if index not in junction_indices:
            continue
        x = 0.05 + 0.90 * junction.toehold_span.start / target_length
        width = 0.90 * (junction.toehold_span.end - junction.toehold_span.start) / target_length
        axis.add_patch(Rectangle((x, y - 0.08), width, 0.16, facecolor=_TOEHOLD, edgecolor="white", linewidth=0.8))
        axis.plot([x + width / 2, x + width / 2], [y - 0.16, y + 0.16], color=_INK, linewidth=0.7)
    axis.text(0.05, 0.32, "0", fontsize=7.5, color=_MUTED, ha="center")
    axis.text(0.95, 0.32, str(target_length), fontsize=7.5, color=_MUTED, ha="center")
    if len(fragment_indices) < len(review.geometry.fragments):
        axis.text(
            0.02,
            0.13,
            f"Geometry preview · {len(fragment_indices)}/{len(review.geometry.fragments)} fragments · "
            f"{len(junction_indices)}/{len(review.geometry.junctions)} junctions",
            fontsize=6.8,
            color=_MUTED,
        )
    else:
        axis.text(0.02, 0.13, "Gray: target domains", fontsize=7.6, color=_MUTED)
        axis.text(0.54, 0.13, "Blue: selected toeholds", fontsize=7.6, color=_MUTED)


def _panel_junctions(axis, review: ThreeWayJunctionReviewV1) -> None:
    junctions = review.geometry.junctions
    axis.text(
        0.02,
        0.92,
        f"{len(junctions)} target junction{'s' if len(junctions) != 1 else ''} · "
        f"{'every target junction shown' if len(junctions) <= 5 else 'bounded target-junction preview'}",
        fontsize=8.5,
        color=_MUTED,
        va="top",
    )
    if len(junctions) <= 5:
        axis.text(
            0.02,
            0.83,
            "Sequence previews · digest = SHA-256[:12]",
            fontsize=6.5,
            color=_MUTED,
            va="top",
        )
        axis.text(0.02, 0.77, "T = toehold · B = matched barcode", fontsize=6.5, color=_MUTED, va="top")
        row_gap = min(0.125, 0.58 / max(1, len(junctions)))
        for index, junction in enumerate(junctions):
            row_y = 0.68 - index * row_gap
            toehold_preview = bounded_sequence_preview(junction.toehold)
            barcode_preview = bounded_sequence_preview(junction.barcode)
            axis.scatter([0.04], [row_y + 0.028], s=22, color=_TOEHOLD, edgecolors="white", linewidths=0.5)
            axis.scatter([0.04], [row_y - 0.028], s=22, color=_BARCODE, edgecolors="white", linewidths=0.5)
            axis.text(
                0.07,
                row_y + 0.028,
                toehold_preview.label(f"J{index + 1} T"),
                fontsize=6.2,
                family="monospace",
                color=_INK,
                va="center",
            )
            axis.text(
                0.07,
                row_y - 0.028,
                barcode_preview.label(f"J{index + 1} B"),
                fontsize=6.2,
                family="monospace",
                color=_INK,
                va="center",
            )
    else:
        y = 0.74
        displayed_indices = _display_indices(len(junctions))
        row_gap = min(0.11, 0.55 / len(displayed_indices))
        for display_index, index in enumerate(displayed_indices):
            row_y = y - display_index * row_gap
            axis.scatter([0.09], [row_y], s=28, color=_TOEHOLD, edgecolors="white", linewidths=0.6, zorder=3)
            axis.scatter([0.54], [row_y], s=28, color=_BARCODE, edgecolors="white", linewidths=0.6, zorder=3)
            axis.plot([0.11, 0.52], [row_y, row_y], color=_GRID, linewidth=1.0, zorder=1)
            axis.text(0.14, row_y, f"J{index + 1}", fontsize=7.6, color=_INK, va="center")
            axis.text(0.59, row_y, f"J{index + 1}", fontsize=7.6, color=_INK, va="center")
        axis.text(0.03, 0.18, "toehold", fontsize=7.8, color=_TOEHOLD, fontweight="semibold")
        axis.text(0.54, 0.18, "matched barcode", fontsize=7.8, color=_BARCODE, fontweight="semibold")
        omitted = len(junctions) - len(displayed_indices)
        axis.text(
            0.03,
            0.08,
            f"{omitted} target junctions omitted from preview; exact assignments remain in the typed contract.",
            fontsize=6.5,
            color=_MUTED,
        )


def _panel_strands_and_recovery(axis, review: ThreeWayJunctionReviewV1) -> None:
    role_counts = {role: 0 for role in ("first", "internal", "last")}
    for strand in review.strands:
        role_counts[strand.role] += 1
    axis.text(
        0.02,
        0.92,
        f"{len(review.strands)} paired-fragment records validated",
        fontsize=8.5,
        color=_MUTED,
        va="top",
    )
    labels = [("first", _TOEHOLD), ("internal", _FRAGMENT), ("last", _BARCODE)]
    for index, (role, color) in enumerate(labels):
        x = 0.03 + index * 0.31
        axis.add_patch(
            FancyBboxPatch(
                (x, 0.66),
                0.27,
                0.11,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor=color,
                edgecolor="none",
                alpha=0.9,
            )
        )
        axis.text(x + 0.135, 0.715, f"{role} · {role_counts[role]}", ha="center", va="center", fontsize=7.5, color=_INK)
    recovery = review.recovery
    axis.text(0.03, 0.53, f"Recovery · {recovery.mode}", fontsize=8.5, fontweight="semibold", color=_RECOVERY)
    axis.text(0.03, 0.46, "Primer sequence previews · digest = SHA-256[:12]", fontsize=6.8, color=_MUTED)
    primer_rows = (
        ("FWD bind", recovery.forward.binding_sequence_5to3),
        ("FWD 5′ ext", recovery.forward.five_prime_extension_5to3),
        ("REV bind", recovery.reverse.binding_sequence_5to3),
        ("REV 5′ ext", recovery.reverse.five_prime_extension_5to3),
    )
    for index, (label, sequence) in enumerate(primer_rows):
        preview = bounded_sequence_preview(sequence)
        axis.text(
            0.03,
            0.39 - index * 0.065,
            preview.label(label),
            fontsize=6.5,
            family="monospace",
            color=_INK,
        )
    axis.text(0.03, 0.10, "Order = declared 5′ extension + binding sequence", fontsize=6.8, color=_MUTED)
    axis.text(0.03, 0.03, "Exact sequences remain in the typed review contract.", fontsize=6.8, color=_MUTED)


def _panel_search_and_checks(axis, review: ThreeWayJunctionReviewV1) -> None:
    search = review.search
    assembly_group_id = bounded_text_preview(search.assembly_group_id)
    if assembly_group_id.abbreviated:
        axis.text(
            0.02,
            0.92,
            (
                f"Assembly group ID · {assembly_group_id.length_chars} chars · "
                f"SHA-256[:12] {assembly_group_id.sha256_prefix}"
            ),
            fontsize=6.5,
            fontweight="semibold",
            color=_INK,
            va="top",
        )
        axis.text(
            0.02, 0.86, f"preview {assembly_group_id.preview}", fontsize=6.5, family="monospace", color=_INK, va="top"
        )
        metrics_y = 0.70
    else:
        axis.text(
            0.02,
            0.92,
            f"Assembly group {assembly_group_id.preview}",
            fontsize=9.0,
            fontweight="semibold",
            color=_INK,
            va="top",
        )
        metrics_y = 0.81
    locus_count, locus_abbreviated = _integer_preview(search.locus_count)
    toehold_paths, paths_abbreviated = _integer_preview(search.toehold_paths_evaluated)
    barcode_candidates, candidates_abbreviated = _integer_preview(search.barcode_candidates_generated)
    barcode_subsets, subsets_abbreviated = _integer_preview(search.barcode_subsets_evaluated)
    matchings, matchings_abbreviated = _integer_preview(search.matchings_evaluated)
    metrics = (
        f"assembly-group loci  {locus_count}",
        f"toehold paths  {toehold_paths}",
        f"toehold min / mean  {search.toehold_min_distance:g} / {search.toehold_mean_distance:g}",
        f"barcode candidates  {barcode_candidates}",
        f"barcode subsets  {barcode_subsets}",
        f"barcode min / mean  {search.barcode_min_distance:g} / {search.barcode_mean_distance:g}",
        f"matchings  {matchings}",
    )
    if any((locus_abbreviated, paths_abbreviated, candidates_abbreviated, subsets_abbreviated, matchings_abbreviated)):
        axis.text(
            0.02,
            0.78,
            "Large integers · digest = SHA-256(unsigned big-endian)[:12]",
            fontsize=5.8,
            color=_MUTED,
        )
    for index, line in enumerate(metrics):
        axis.text(0.03, metrics_y - index * 0.075, line, fontsize=6.5, color=_INK, family="monospace")
    passed = sum(check.status == "passed" for check in review.checks)
    not_run = sum(check.status == "not_run" for check in review.checks)
    axis.text(0.03, 0.25, f"Checks  {passed} passed  ·  {not_run} not run", fontsize=7.8, color=_MUTED)
    axis.add_patch(
        FancyBboxPatch(
            (0.02, 0.06),
            0.94,
            0.11,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=_WARNING_BG,
            edgecolor=_WARNING,
            linewidth=0.8,
        )
    )
    axis.text(
        0.49,
        0.115,
        "THERMODYNAMIC SCREENING NOT RUN",
        ha="center",
        va="center",
        fontsize=7.7,
        fontweight="semibold",
        color=_WARNING,
    )


@dataclass(frozen=True)
class ThreeWayJunctionReviewRenderer:
    def preflight(self, record: Record, style: Style, palette: Palette) -> None:
        _ = palette
        _review_figure_size(style)
        _review_from_record(record)

    def render(self, record: Record, style: Style, palette: Palette):
        _ = palette
        figure_size = _review_figure_size(style)
        review = _review_from_record(record)
        figure, axes = plt.subplots(
            1,
            4,
            figsize=figure_size,
            dpi=style.dpi,
        )
        titles = ("Target geometry", "Junction assignments", "Strands and recovery", "Search and checks")
        for axis, title, gid in zip(axes, titles, _AXIS_GIDS, strict=True):
            _setup_axis(axis, title=title, gid=gid)
        _panel_geometry(axes[0], review)
        _panel_junctions(axes[1], review)
        _panel_strands_and_recovery(axes[2], review)
        _panel_search_and_checks(axes[3], review)
        figure.suptitle(
            "Three-way-junction design review",
            x=0.02,
            y=0.99,
            ha="left",
            fontsize=13,
            fontweight="semibold",
            color=_INK,
        )
        figure.subplots_adjust(left=0.025, right=0.99, top=0.84, bottom=0.08, wspace=0.12)
        return figure


__all__ = ["ThreeWayJunctionReviewRenderer"]
