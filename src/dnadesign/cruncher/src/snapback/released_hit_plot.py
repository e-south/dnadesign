"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_hit_plot.py

Truthful origin-anchored plotting for released-product snapback solve hits.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from math import cos, radians, sin
from pathlib import Path
from typing import Any

from dnadesign.cruncher.snapback.publication_support import complement_sequence
from dnadesign.cruncher.snapback.released_models import ReleasedTargetSearchHit

_TEXT = "#334155"
_TITLE = "#475569"
_BOUNDARY = "#0F172A"
_NICK = "#2563EB"
_RELEASE = "#D97706"
_RETAINED = "#059669"
_ACTIVE_BOTTOM = "#0F766E"
_TAIL = "#94A3B8"
_STEM = "#DC2626"
_CAP = "#DB2777"
_FOLDBACK = "#7C3AED"
_OVERHANG = "#64748B"


def _span(start: int, end: int) -> dict[str, int]:
    return {"start": start, "end": end}


def _watson_crick_mismatch_positions(*, top_sequence: str, bottom_sequence: str) -> list[int]:
    mismatches: list[int] = []
    for index, (top_base, bottom_base) in enumerate(zip(top_sequence, bottom_sequence, strict=True)):
        if complement_sequence(bottom_base) != top_base:
            mismatches.append(index)
    return mismatches


def build_released_hit_plot_context(hit: ReleasedTargetSearchHit) -> dict[str, Any]:
    if hit.pre_nick_site.local_start is not None and hit.pre_nick_site.local_start < 0:
        raise ValueError("Released solve plot requires non-negative pre-nick site local coordinates.")
    if hit.release_site.local_start is not None and hit.release_site.local_start < 0:
        raise ValueError("Released solve plot requires non-negative release site local coordinates.")
    active_bottom_length = hit.projection.active_bottom_length_nt
    retained_top_length = hit.projection.retained_top_length_nt
    paired_bp = hit.final_candidate.paired_bp
    cap_nt = hit.final_candidate.cap_nt
    structure_width = (2 * paired_bp) + cap_nt
    structure_start = active_bottom_length - structure_width
    if structure_start < 0:
        raise ValueError(
            "Released solve plot requires active_bottom_length_nt >= (2 * paired_bp) + cap_nt for bottom-strand view."
        )
    stem_span = _span(structure_start, structure_start + paired_bp)
    cap_span = _span(stem_span["end"], stem_span["end"] + cap_nt)
    foldback_span = _span(cap_span["end"], cap_span["end"] + paired_bp)
    precursor_top_span = _span(0, len(hit.precursor_top_strand))
    precursor_bottom_span = _span(0, len(hit.precursor_top_strand))
    retained_top_span = _span(0, retained_top_length)
    active_bottom_span = _span(0, active_bottom_length)
    duplex_overlap_end = min(retained_top_length, active_bottom_length)
    duplex_overlap_span = _span(0, duplex_overlap_end) if duplex_overlap_end > 0 else None
    top_only_overhang_span = (
        _span(active_bottom_span["end"], retained_top_span["end"])
        if retained_top_span["end"] > active_bottom_span["end"]
        else None
    )
    bottom_only_overhang_span = (
        _span(retained_top_span["end"], active_bottom_span["end"])
        if active_bottom_span["end"] > retained_top_span["end"]
        else None
    )
    duplex_top_sequence = hit.projection.retained_top_strand[:duplex_overlap_end] if duplex_overlap_end > 0 else ""
    duplex_bottom_sequence = hit.projection.active_bottom_strand[:duplex_overlap_end]
    foldback_sequence = hit.projection.active_bottom_strand[foldback_span["start"] : foldback_span["end"]]
    foldback_partner_sequence = foldback_sequence[::-1]
    return {
        "kind": "released_hit_plot_v1",
        "target": {
            "nick_boundary_from_left": hit.nick_boundary_from_left,
            "paired_bp": paired_bp,
            "cap_nt": cap_nt,
        },
        "nickase_variant_id": hit.nickase_variant_id,
        "release_variant_id": hit.release_variant_id,
        "precursor": {
            "top_sequence": hit.precursor_top_strand,
            "bottom_sequence": complement_sequence(hit.precursor_top_strand),
            "nick_site": hit.pre_nick_site.model_dump(mode="json"),
            "nick_event": hit.pre_nick_event.model_dump(mode="json"),
            "release_site": hit.release_site.model_dump(mode="json"),
            "release_event": hit.release_event.model_dump(mode="json"),
            "top_span": precursor_top_span,
            "bottom_span": precursor_bottom_span,
            "retained_top_span": retained_top_span,
            "active_bottom_span": active_bottom_span,
            "sacrificial_top_tail_span": _span(retained_top_length, len(hit.precursor_top_strand)),
            "sacrificial_bottom_tail_span": _span(active_bottom_length, len(hit.precursor_top_strand)),
        },
        "released_product": {
            "retained_top_sequence": hit.projection.retained_top_strand,
            "active_bottom_sequence": hit.projection.active_bottom_strand,
            "nick_origin_boundary": hit.nick_boundary_from_left,
            "release_top_cut_boundary": hit.projection.release_top_cut_precursor,
            "release_bottom_cut_boundary": hit.projection.release_bottom_cut_precursor,
            "retained_top_span": retained_top_span,
            "active_bottom_span": active_bottom_span,
            "duplex_overlap_span": duplex_overlap_span,
            "duplex_top_sequence": duplex_top_sequence,
            "duplex_bottom_sequence": duplex_bottom_sequence,
            "duplex_mismatch_positions": _watson_crick_mismatch_positions(
                top_sequence=duplex_top_sequence,
                bottom_sequence=duplex_bottom_sequence,
            ),
            "top_only_overhang_span": top_only_overhang_span,
            "bottom_only_overhang_span": bottom_only_overhang_span,
            "active_prefix_span": _span(0, structure_start),
            "stem_span": stem_span,
            "cap_span": cap_span,
            "foldback_span": foldback_span,
            "nickase_site_survives_post_release": hit.projection.nickase_site_survives_post_release,
            "release_site_survives_post_release": hit.projection.release_site_survives_post_release,
        },
        "foldback": {
            "origin_boundary_from_left": stem_span["start"],
            "stem_sequence": hit.projection.active_bottom_strand[stem_span["start"] : stem_span["end"]],
            "cap_sequence": hit.projection.active_bottom_strand[cap_span["start"] : cap_span["end"]],
            "foldback_sequence": foldback_sequence,
            "foldback_partner_sequence": foldback_partner_sequence,
            "foldback_mismatch_positions": _watson_crick_mismatch_positions(
                top_sequence=hit.projection.active_bottom_strand[stem_span["start"] : stem_span["end"]],
                bottom_sequence=foldback_partner_sequence,
            ),
        },
    }


def _draw_sequence(
    ax,
    *,
    sequence: str,
    y: float,
    row_label: str,
    start_terminal: str,
    end_terminal: str,
    x_start: float = 0.0,
) -> None:
    ax.text(x_start - 1.0, y, row_label, ha="right", va="center", fontsize=14, family="DejaVu Sans", color=_TEXT)
    ax.text(x_start + 0.06, y, start_terminal, ha="right", va="center", fontsize=12, family="DejaVu Sans", color=_TITLE)
    ax.text(
        x_start + len(sequence) + 0.94,
        y,
        end_terminal,
        ha="left",
        va="center",
        fontsize=12,
        family="DejaVu Sans",
        color=_TITLE,
    )
    for index, base in enumerate(sequence):
        ax.text(
            x_start + index + 0.5,
            y,
            base,
            ha="center",
            va="center",
            fontsize=19,
            family="DejaVu Sans Mono",
            color=_TEXT,
        )


def _draw_span(ax, *, start: int, end: int, y: float, label: str, color: str, label_above: bool = True) -> None:
    if end <= start:
        return
    x0 = start + 0.08
    x1 = end - 0.08
    ax.plot([x0, x1], [y, y], color=color, linewidth=2.2, solid_capstyle="round")
    ax.plot([x0, x0], [y - 0.05, y + 0.05], color=color, linewidth=1.2)
    ax.plot([x1, x1], [y - 0.05, y + 0.05], color=color, linewidth=1.2)
    ax.text(
        (start + end) / 2.0,
        y + 0.15 if label_above else y - 0.16,
        label,
        ha="center",
        va="bottom" if label_above else "top",
        fontsize=10.5,
        family="DejaVu Sans",
        color=color,
    )


def _draw_boundary(
    ax,
    *,
    boundary: int,
    y0: float,
    y1: float,
    label: str,
    color: str = _BOUNDARY,
    dashed: bool = False,
    label_y: float | None = None,
) -> None:
    line = ax.plot([boundary, boundary], [y0, y1], color=color, linewidth=1.2)[0]
    if dashed:
        line.set_dashes((2.5, 2.0))
    else:
        ax.plot([boundary - 0.08, boundary + 0.08], [y1, y1], color=color, linewidth=1.0)
    ax.text(
        boundary,
        label_y if label_y is not None else y1 + 0.12,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        family="DejaVu Sans",
        color=color,
    )


def _configure_axis(ax, *, x_min: float, x_max: float, title: str) -> None:
    ax.set_axis_off()
    ax.set_xlim(x_min - 3.5, x_max + 2.4)
    ax.set_ylim(-1.1, 2.72)
    ax.text(x_min - 3.2, 2.48, title, ha="left", va="top", fontsize=17, family="DejaVu Sans", color=_TITLE)


def _render_precursor_panel(ax, context: dict[str, Any]) -> None:
    precursor = context["precursor"]
    top = precursor["top_sequence"]
    bottom = precursor["bottom_sequence"]
    top_span = precursor["top_span"]
    bottom_span = precursor["bottom_span"]
    _configure_axis(
        ax,
        x_min=min(top_span["start"], bottom_span["start"]),
        x_max=max(top_span["end"], bottom_span["end"]),
        title="precursor sites",
    )
    _draw_sequence(
        ax,
        sequence=top,
        y=1.15,
        row_label="Top",
        start_terminal="5'",
        end_terminal="3'",
        x_start=top_span["start"],
    )
    _draw_sequence(
        ax,
        sequence=bottom,
        y=0.25,
        row_label="Bottom",
        start_terminal="3'",
        end_terminal="5'",
        x_start=bottom_span["start"],
    )
    for index in range(len(top)):
        x = top_span["start"] + index + 0.5
        ax.plot([x, x], [0.37, 1.03], color="#E2E8F0", linewidth=0.8)
    _draw_boundary(
        ax,
        boundary=precursor["nick_event"]["boundary"],
        y0=0.08,
        y1=1.32,
        label="Nick / origin",
        color=_NICK,
    )
    _draw_boundary(
        ax,
        boundary=precursor["release_event"]["top_cut_boundary"],
        y0=0.08,
        y1=1.32,
        label="Top cut",
        color=_RELEASE,
        label_y=1.48,
    )
    if precursor["release_event"]["bottom_cut_boundary"] != precursor["release_event"]["top_cut_boundary"]:
        _draw_boundary(
            ax,
            boundary=precursor["release_event"]["bottom_cut_boundary"],
            y0=-0.02,
            y1=0.42,
            label="Bottom cut",
            color=_RELEASE,
            dashed=True,
            label_y=-0.18,
        )
    _draw_span(
        ax,
        start=precursor["nick_site"]["local_start"],
        end=precursor["nick_site"]["local_end"],
        y=1.72,
        label="nickase site",
        color=_NICK,
    )
    _draw_span(
        ax,
        start=precursor["release_site"]["local_start"],
        end=precursor["release_site"]["local_end"],
        y=2.02,
        label="release site",
        color=_RELEASE,
    )
    _draw_span(
        ax,
        start=precursor["retained_top_span"]["start"],
        end=precursor["retained_top_span"]["end"],
        y=-0.34,
        label="retained top",
        color=_RETAINED,
        label_above=False,
    )
    _draw_span(
        ax,
        start=precursor["active_bottom_span"]["start"],
        end=precursor["active_bottom_span"]["end"],
        y=-0.64,
        label="active bottom",
        color=_ACTIVE_BOTTOM,
        label_above=False,
    )
    _draw_span(
        ax,
        start=precursor["sacrificial_top_tail_span"]["start"],
        end=precursor["sacrificial_top_tail_span"]["end"],
        y=-0.34,
        label="top tail",
        color=_TAIL,
        label_above=False,
    )
    _draw_span(
        ax,
        start=precursor["sacrificial_bottom_tail_span"]["start"],
        end=precursor["sacrificial_bottom_tail_span"]["end"],
        y=-0.64,
        label="bottom tail",
        color=_TAIL,
        label_above=False,
    )


def _render_released_panel(ax, context: dict[str, Any]) -> None:
    released_product = context["released_product"]
    top = released_product["retained_top_sequence"]
    bottom = released_product["active_bottom_sequence"]
    top_span = released_product["retained_top_span"]
    bottom_span = released_product["active_bottom_span"]
    _configure_axis(
        ax,
        x_min=min(top_span["start"], bottom_span["start"]),
        x_max=max(top_span["end"], bottom_span["end"]),
        title="post-release fragments",
    )
    _draw_sequence(
        ax,
        sequence=top,
        y=1.15,
        row_label="Top fragment",
        start_terminal="5'",
        end_terminal="3'",
        x_start=top_span["start"],
    )
    _draw_sequence(
        ax,
        sequence=bottom,
        y=0.25,
        row_label="Active bottom",
        start_terminal="3'",
        end_terminal="5'",
        x_start=bottom_span["start"],
    )
    duplex_overlap = released_product["duplex_overlap_span"]
    duplex_mismatches = set(released_product["duplex_mismatch_positions"])
    if duplex_overlap is not None:
        for local_index in range(duplex_overlap["start"], duplex_overlap["end"]):
            color = "#CBD5E1" if local_index not in duplex_mismatches else _STEM
            ax.plot([local_index + 0.5, local_index + 0.5], [0.37, 1.03], color=color, linewidth=0.8)
    _draw_boundary(
        ax,
        boundary=released_product["nick_origin_boundary"],
        y0=0.08,
        y1=1.32,
        label="Nick / origin",
        color=_NICK,
        label_y=1.48,
    )
    _draw_boundary(
        ax,
        boundary=released_product["release_top_cut_boundary"],
        y0=0.86,
        y1=1.32,
        label="Top cut",
        color=_RELEASE,
        dashed=True,
    )
    _draw_boundary(
        ax,
        boundary=released_product["release_bottom_cut_boundary"],
        y0=0.08,
        y1=0.54,
        label="Bottom cut",
        color=_RELEASE,
        dashed=True,
        label_y=-0.18,
    )
    _draw_span(
        ax,
        start=released_product["active_prefix_span"]["start"],
        end=released_product["active_prefix_span"]["end"],
        y=1.72,
        label="prefix",
        color=_TAIL,
    )
    _draw_span(
        ax,
        start=released_product["stem_span"]["start"],
        end=released_product["stem_span"]["end"],
        y=2.02,
        label="stem",
        color=_STEM,
    )
    _draw_span(
        ax,
        start=released_product["cap_span"]["start"],
        end=released_product["cap_span"]["end"],
        y=1.72,
        label="cap",
        color=_CAP,
    )
    _draw_span(
        ax,
        start=released_product["foldback_span"]["start"],
        end=released_product["foldback_span"]["end"],
        y=2.02,
        label="foldback",
        color=_FOLDBACK,
    )
    top_only_overhang = released_product["top_only_overhang_span"]
    if top_only_overhang is not None:
        _draw_span(
            ax,
            start=top_only_overhang["start"],
            end=top_only_overhang["end"],
            y=-0.34,
            label="top-only overhang",
            color=_OVERHANG,
            label_above=False,
        )
    bottom_only_overhang = released_product["bottom_only_overhang_span"]
    if bottom_only_overhang is not None:
        _draw_span(
            ax,
            start=bottom_only_overhang["start"],
            end=bottom_only_overhang["end"],
            y=-0.64,
            label="bottom-only overhang",
            color=_OVERHANG,
            label_above=False,
        )
    ax.text(
        min(top_span["start"], bottom_span["start"]) - 3.2,
        -0.86,
        (
            "nickase site survives: "
            f"{'yes' if released_product['nickase_site_survives_post_release'] else 'no'}"
            "   release site survives: "
            f"{'yes' if released_product['release_site_survives_post_release'] else 'no'}"
        ),
        ha="left",
        va="bottom",
        fontsize=10,
        family="DejaVu Sans",
        color=_TITLE,
    )


def _render_foldback_panel(ax, context: dict[str, Any]) -> None:
    from matplotlib.patches import Arc

    foldback = context["foldback"]
    stem = foldback["stem_sequence"]
    foldback_partner = foldback["foldback_partner_sequence"]
    foldback_mismatches = set(foldback["foldback_mismatch_positions"])
    cap_sequence = foldback["cap_sequence"]
    origin_x = len(stem) + 1.45

    ax.set_axis_off()
    ax.set_xlim(-1.7, max(8.5, origin_x + 1.4))
    ax.set_ylim(-0.8, 2.65)
    ax.text(
        -1.4,
        2.24,
        "origin-anchored bottom foldback",
        ha="left",
        va="top",
        fontsize=17,
        family="DejaVu Sans",
        color=_TITLE,
    )

    ax.text(-0.7, 1.22, "Active stem", ha="right", va="center", fontsize=14, family="DejaVu Sans", color=_TEXT)
    ax.text(-0.7, 0.3, "Foldback return", ha="right", va="center", fontsize=14, family="DejaVu Sans", color=_TEXT)
    ax.text(0.06, 1.22, "3'", ha="right", va="center", fontsize=12, family="DejaVu Sans", color=_TITLE)
    ax.text(0.06, 0.3, "5'", ha="right", va="center", fontsize=12, family="DejaVu Sans", color=_TITLE)

    for index, base in enumerate(stem):
        x = index + 0.5
        ax.text(x, 1.22, base, ha="center", va="center", fontsize=19, family="DejaVu Sans Mono", color=_TEXT)
        ax.text(
            x,
            0.3,
            foldback_partner[index],
            ha="center",
            va="center",
            fontsize=19,
            family="DejaVu Sans Mono",
            color=_TEXT,
        )
        ax.plot([x, x], [0.42, 1.04], color="#CBD5E1" if index not in foldback_mismatches else _STEM, linewidth=0.9)

    arc_center_x = len(stem) + 0.75
    arc_width = 1.2
    arc_height = 1.3
    ax.add_patch(
        Arc((arc_center_x, 0.76), width=arc_width, height=arc_height, theta1=-90, theta2=90, color=_CAP, linewidth=1.6)
    )
    if cap_sequence:
        theta_values = [65.0]
        if len(cap_sequence) > 1:
            theta_values = [65.0 - ((130.0 / (len(cap_sequence) - 1)) * index) for index in range(len(cap_sequence))]
        radius_x = arc_width / 2.0
        radius_y = arc_height / 2.0
        for base, theta_deg in zip(cap_sequence, theta_values, strict=True):
            theta = radians(theta_deg)
            x = arc_center_x + (radius_x * cos(theta))
            y = 0.76 + (radius_y * sin(theta))
            ax.text(
                x,
                y,
                base,
                ha="center",
                va="center",
                fontsize=16,
                family="DejaVu Sans Mono",
                color=_CAP,
            )
    _draw_boundary(
        ax,
        boundary=0,
        y0=0.12,
        y1=1.46,
        label="Nick / origin",
        color=_NICK,
        label_y=1.62,
    )


def render_released_hit_plot(hit: ReleasedTargetSearchHit, output_path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    context = build_released_hit_plot_context(hit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_width = max(10.5, len(hit.precursor_top_strand) * 0.48)
    fig, axes = plt.subplots(3, 1, figsize=(figure_width, 10.6), dpi=160)
    _render_precursor_panel(axes[0], context)
    _render_released_panel(axes[1], context)
    _render_foldback_panel(axes[2], context)
    fig.tight_layout(h_pad=1.4)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return context


__all__ = ["build_released_hit_plot_context", "render_released_hit_plot"]
