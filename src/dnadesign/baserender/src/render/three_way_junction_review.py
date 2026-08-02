"""Four-panel QA renderer for neutral three-way-junction review evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from pydantic import ValidationError

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from ..config import Style
from ..core import Record, RenderingError
from ..core.pydantic_validation import format_validation_error
from .palette import Palette

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
    axis.text(0.02, 0.92, review.target.target_id, fontsize=10, fontweight="semibold", color=_INK, va="top")
    axis.text(
        0.02,
        0.83,
        f"{target_length} nt  ·  {len(review.geometry.fragments)} fragments  ·  {len(review.geometry.junctions)} "
        f"junction{'s' if len(review.geometry.junctions) != 1 else ''}",
        fontsize=8.2,
        color=_MUTED,
        va="top",
    )
    y = 0.55
    axis.plot([0.05, 0.95], [y, y], color=_INK, linewidth=1.2, zorder=1)
    for fragment in review.geometry.fragments:
        x = 0.05 + 0.90 * fragment.domain_span.start / target_length
        width = 0.90 * (fragment.domain_span.end - fragment.domain_span.start) / target_length
        axis.add_patch(Rectangle((x, y - 0.08), width, 0.16, facecolor=_FRAGMENT, edgecolor="white", linewidth=1.0))
        axis.text(x + width / 2, y, f"F{fragment.index + 1}", ha="center", va="center", fontsize=8, color=_INK)
    for junction in review.geometry.junctions:
        x = 0.05 + 0.90 * junction.toehold_span.start / target_length
        width = 0.90 * (junction.toehold_span.end - junction.toehold_span.start) / target_length
        axis.add_patch(Rectangle((x, y - 0.08), width, 0.16, facecolor=_TOEHOLD, edgecolor="white", linewidth=0.8))
        axis.plot([x + width / 2, x + width / 2], [y - 0.16, y + 0.16], color=_INK, linewidth=0.7)
    axis.text(0.05, 0.32, "0", fontsize=7.5, color=_MUTED, ha="center")
    axis.text(0.95, 0.32, str(target_length), fontsize=7.5, color=_MUTED, ha="center")
    axis.text(0.02, 0.13, "Gray: target domains", fontsize=7.6, color=_MUTED)
    axis.text(0.54, 0.13, "Blue: selected toeholds", fontsize=7.6, color=_MUTED)


def _panel_junctions(axis, review: ThreeWayJunctionReviewV1) -> None:
    junctions = review.geometry.junctions
    axis.text(
        0.02,
        0.92,
        f"{len(junctions)} junction{'s' if len(junctions) != 1 else ''} · every locus shown",
        fontsize=8.5,
        color=_MUTED,
        va="top",
    )
    y = 0.74
    row_gap = min(0.14, 0.55 / max(1, len(junctions)))
    for index, junction in enumerate(junctions):
        row_y = y - index * row_gap
        axis.scatter([0.09], [row_y], s=28, color=_TOEHOLD, edgecolors="white", linewidths=0.6, zorder=3)
        axis.scatter([0.54], [row_y], s=28, color=_BARCODE, edgecolors="white", linewidths=0.6, zorder=3)
        axis.plot([0.11, 0.52], [row_y, row_y], color=_GRID, linewidth=1.0, zorder=1)
        if len(junctions) <= 5 or index in {0, len(junctions) - 1}:
            axis.text(0.14, row_y, junction.toehold, fontsize=7.6, family="monospace", color=_INK, va="center")
            axis.text(0.59, row_y, junction.barcode, fontsize=7.6, family="monospace", color=_INK, va="center")
        else:
            axis.text(0.14, row_y, f"J{index + 1}", fontsize=7.6, color=_INK, va="center")
            axis.text(0.59, row_y, f"J{index + 1}", fontsize=7.6, color=_INK, va="center")
    axis.text(0.03, 0.18, "toehold", fontsize=7.8, color=_TOEHOLD, fontweight="semibold")
    axis.text(0.54, 0.18, "matched barcode", fontsize=7.8, color=_BARCODE, fontweight="semibold")
    if len(junctions) > 5:
        axis.text(0.03, 0.08, "Interior sequences remain explicit in the review contract.", fontsize=7.0, color=_MUTED)


def _extension_label(sequence: str) -> str:
    return sequence if sequence else "none"


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
    axis.text(
        0.03,
        0.42,
        f"Forward  bind {recovery.forward.binding_sequence_5to3}  ·  5′ extension "
        f"{_extension_label(recovery.forward.five_prime_extension_5to3)}",
        fontsize=7.3,
        family="monospace",
        color=_INK,
    )
    axis.text(
        0.03,
        0.31,
        f"Reverse  bind {recovery.reverse.binding_sequence_5to3}  ·  5′ extension "
        f"{_extension_label(recovery.reverse.five_prime_extension_5to3)}",
        fontsize=7.3,
        family="monospace",
        color=_INK,
    )
    axis.text(0.03, 0.15, "Order sequence = declared extension + binding sequence", fontsize=7.2, color=_MUTED)
    axis.text(0.03, 0.07, "No cloning meaning is inferred from an extension.", fontsize=7.2, color=_MUTED)


def _panel_search_and_checks(axis, review: ThreeWayJunctionReviewV1) -> None:
    search = review.search
    axis.text(0.02, 0.92, f"Pool {search.pool_id}", fontsize=9.0, fontweight="semibold", color=_INK, va="top")
    metrics = (
        f"loci  {search.locus_count}",
        f"toehold paths  {search.toehold_paths_evaluated}",
        f"toehold min / mean  {search.toehold_min_distance:g} / {search.toehold_mean_distance:g}",
        f"barcode candidates  {search.barcode_candidates_generated}",
        f"barcode subsets  {search.barcode_subsets_evaluated}",
        f"barcode min / mean  {search.barcode_min_distance:g} / {search.barcode_mean_distance:g}",
        f"matchings  {search.matchings_evaluated}",
    )
    for index, line in enumerate(metrics):
        axis.text(0.03, 0.81 - index * 0.075, line, fontsize=7.5, color=_INK, family="monospace")
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
    def render(self, record: Record, style: Style, palette: Palette):
        _ = palette
        review = _review_from_record(record)
        figure, axes = plt.subplots(
            1,
            4,
            figsize=(15.2 * float(style.figure_scale), 4.2 * float(style.figure_scale)),
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
