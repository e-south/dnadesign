"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/render.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from .layout import (
    assign_tracks,
    assign_tracks_forward_with_sigma_lock,
    comp,
    measure_char_cell,
)
from .model import Guide, SeqRecord
from .palette import Palette
from .style import Style


@dataclass
class LayoutResult:
    cw: float
    ch: float
    width: float
    height: float
    y_forward: float
    y_reverse: float
    up_tracks: Sequence[int]
    dn_tracks: Sequence[int]
    n_up_tracks: int
    n_dn_tracks: int
    x_left: float  # base x for column 0 (includes label margin)


def _text_px_width(text: str, family: str, size_pt: int, dpi: int) -> float:
    prop = FontProperties(family=family, size=size_pt)
    tp = TextPath((0, 0), text, prop=prop)
    bbox = tp.get_extents()
    return bbox.width / 72.0 * dpi


def _compute_layout(
    record: SeqRecord,
    style: Style,
    *,
    fixed_tracks: tuple[int, int] | None = None,
    fixed_n: Optional[int] = None,
) -> LayoutResult:
    size = measure_char_cell(style.font_mono, style.font_size_seq, style.dpi)
    cw, ch = size.width, size.height
    # Actual box (k-mer) height used for vertical budgeting
    h = ch * style.kmer.height_factor
    n = fixed_n if fixed_n is not None else len(record.sequence)

    # Dynamic horizontal padding so 5′/3′ labels never clip
    label_pad_x = style.font_size_label / 72.0 * style.dpi * 1.6
    x_left = style.padding_x + label_pad_x

    # Track counts (honor priority via assign_tracks)
    up = [a for a in record.annotations if a.strand == "fwd"]
    dn = [a for a in record.annotations if a.strand == "rev"]
    up_tracks = assign_tracks_forward_with_sigma_lock(up)
    dn_tracks = assign_tracks(dn)
    used_up = (max(up_tracks) + 1) if up_tracks else 0
    used_dn = (max(dn_tracks) + 1) if dn_tracks else 0
    count_up = fixed_tracks[0] if fixed_tracks else used_up
    count_dn = fixed_tracks[1] if fixed_tracks else used_dn

    # Vertical extents (baselines)
    y0 = style.padding_y + count_up * style.track_spacing + ch
    y1 = y0 - style.baseline_spacing if style.show_reverse_complement else y0

    label_pad_y = style.font_size_label / 72.0 * style.dpi * 1.2
    legend_space = (style.legend_height_px + style.legend_pad_px) if style.legend else 0.0

    # Compute content region independent of legend; legend is appended below.
    # Top-most extent must include *all upward tracks*; previously this was undercounted.
    top = y0 + count_up * style.track_spacing + h + label_pad_y
    bottom = y1 - (count_dn * style.track_spacing) - h - label_pad_y
    # Small safety margin so rounded corners / labels never clip the axes edge.
    margin = max(2.0, 0.5 * style.kmer.round_px)
    top += margin
    bottom -= margin
    # --- Keep the legend area clear by lifting baselines if needed -------------
    # Ensure the entire content block sits ABOVE the reserved legend space
    # [0 .. legend_space]. If bottom < legend_space, shift everything up.
    if legend_space > 0 and bottom < legend_space:
        dy = legend_space - bottom
        y0 += dy
        y1 += dy
        top += dy
        bottom += dy

    content_height = top - bottom + style.padding_y
    height = content_height + legend_space
    width = x_left + n * cw + style.padding_x + label_pad_x

    return LayoutResult(cw, ch, width, height, y0, y1, up_tracks, dn_tracks, count_up, count_dn, x_left)


def _draw_sequence(
    ax,
    s: str,
    x0: float,
    y_center: float,
    cw: float,
    style: Style,
    left_label: str,
    right_label: str,
):
    """
    Draw the sequence row with monospaced, data-space glyphs.

    - Letters are converted to vector paths (TextPath) and added as PathPatch
      with transform=ax.transData.
    - Each glyph is scaled from points → pixels (data units) using dpi/72.
    - Horizontal advance uses the same cw that boxes and guides use, so columns
      stay perfectly aligned in X with k-mers and connectors.
    """
    # Small 5′ / 3′ labels (screen text is fine; they don't need column alignment)
    ax.text(
        x0 - (style.font_size_label / 72.0 * style.dpi * 0.8),
        y_center,
        left_label,
        va="center",
        ha="right",
        fontsize=style.font_size_label,
        family=style.font_label,
        color=style.color_sequence,
        alpha=0.9,
    )
    ax.text(
        x0 + len(s) * cw + (style.font_size_label / 72.0 * style.dpi * 0.4),
        y_center,
        right_label,
        va="center",
        ha="left",
        fontsize=style.font_size_label,
        family=style.font_label,
        color=style.color_sequence,
        alpha=0.9,
    )

    # Data-space glyphs (one shared vertical centering for all letters)
    prop = FontProperties(family=style.font_mono, size=style.font_size_seq)
    px_per_pt = style.dpi / 72.0
    # Compute one font-wide center offset using "Ag" (captures asc/desc)
    _ag = TextPath((0, 0), "Ag", prop=prop, usetex=False).get_extents()
    y_mid_px = ((_ag.y0 + _ag.y1) / 2.0) * px_per_pt
    cache: dict[str, TextPath] = {}

    x = x0
    for ch in s:
        tp = cache.get(ch)
        if tp is None:
            tp = TextPath((0, 0), ch, prop=prop, usetex=False)
            cache[ch] = tp
        trans = Affine2D().scale(px_per_pt).translate(x, y_center - y_mid_px) + ax.transData
        ax.add_patch(
            PathPatch(
                tp,
                transform=trans,
                facecolor=style.color_sequence,
                edgecolor="none",
                linewidth=0.0,
                zorder=2,
                clip_on=False,
            )
        )
        x += cw


def _draw_box(
    ax,
    x,
    y,
    w,
    h,
    label: str,
    facecolor: Tuple[float, float, float],
    style: Style,
    above: bool,
    cw: float,
):
    r = style.kmer.round_px
    pad_x = float(style.kmer.pad_x_px)
    yy = y + (h * 0.5 if above else -h * 0.5)  # centerline of the rounded box
    box = FancyBboxPatch(
        (x - pad_x, yy - h / 2),
        w + 2 * pad_x,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        linewidth=style.kmer.edge_width,
        facecolor=facecolor,
        alpha=style.kmer.fill_alpha,
        edgecolor=facecolor,
        zorder=3,
        clip_on=False,
    )
    ax.add_patch(box)
    # --- Draw k-mer letters in *data space* on the same grid as the reference ---
    # Use the *sequence* font size so glyph metrics match the reference rows.
    prop = FontProperties(family=style.font_mono, size=style.font_size_seq)
    px_per_pt = style.dpi / 72.0
    _ag = TextPath((0, 0), "Ag", prop=prop, usetex=False).get_extents()
    y_mid_px = ((_ag.y0 + _ag.y1) / 2.0) * px_per_pt
    cache: dict[str, TextPath] = {}

    # Column advance MUST equal the global character cell width so glyphs
    # and background boxes stay on the same grid (no fractional drift).
    col_w = cw
    xx = x  # left edge of the k‑mer (box padding handled separately)
    for ch in label:
        tp = cache.get(ch)
        if tp is None:
            tp = TextPath((0, 0), ch, prop=prop, usetex=False)
            cache[ch] = tp
        # Vertical anchor for text: baseline-aligned vs centered in the box.
        align = str(style.kmer.text_v_align).lower()
        y_text_center = yy if align == "center" else y
        y_text_center += float(style.kmer.text_v_offset_px)
        trans = Affine2D().scale(px_per_pt).translate(xx, y_text_center - y_mid_px) + ax.transData
        ax.add_patch(
            PathPatch(
                tp,
                transform=trans,
                facecolor=style.kmer.text_color,
                edgecolor="none",
                linewidth=0.0,
                zorder=4,
                clip_on=False,
            )
        )
        xx += col_w  # advance exactly one column per character


def _draw_guides(
    ax,
    guides: Sequence[Guide],
    x0: float,
    cw: float,
    ch: float,
    y_top: float,
    n_up_tracks: int,
    style: Style,
):
    # Height to draw 'bracket' style (legacy) — kept for backward compatibility
    legacy_y = y_top + (max(1, n_up_tracks) + 0.8) * style.track_spacing

    for g in guides:
        kind = g.kind
        if kind == "bracket":
            x1 = x0 + g.start * cw
            x2 = x0 + g.end * cw
            ax.plot([x1, x2], [legacy_y, legacy_y], color="#9CA3AF", lw=1.5, zorder=2)
            ax.plot([x1, x1], [legacy_y, legacy_y - 6], color="#9CA3AF", lw=1.5, zorder=2)
            ax.plot([x2, x2], [legacy_y, legacy_y - 6], color="#9CA3AF", lw=1.5, zorder=2)
            if g.label:
                ax.text(
                    (x1 + x2) / 2,
                    legacy_y + 6,
                    g.label,
                    ha="center",
                    va="bottom",
                    fontsize=style.font_size_label,
                    family=style.font_label,
                    color="#9CA3AF",
                    zorder=2,
                    clip_on=False,
                )

        elif kind == "sigma_link":
            # Sandwiched span between boxes, drawn at the same track level as sigma boxes
            payload = g.payload or {}
            track = int(payload.get("track", 0))
            up_len = int(payload.get("up_len", 0))
            # Prefer explicit per-guide bp margin, else style default, else legacy frac.
            inner_bp = payload.get("inner_margin_bp", None)
            if inner_bp is not None:
                inner_bp = float(inner_bp)
                inner_px = inner_bp * cw
            else:
                legacy_frac = float(payload.get("inner_margin_frac", -1.0))
                if legacy_frac >= 0:
                    inner_px = legacy_frac * cw
                else:
                    inner_px = float(style.sigma_link_inner_margin_bp) * cw

            # Centerline of sigma box on this track
            h = ch * style.kmer.height_factor
            y_box_center = (y_top + (track + 1) * style.track_spacing) + (h * 0.5)

            # Start after upstream end (+ inner margin), end before downstream start (- margin)
            # line runs *between* boxes: slightly after end(-35) and slightly
            # before start(-10)
            left = (g.start + up_len) * cw + inner_px
            right = (g.end * cw) - inner_px
            x1 = x0 + left
            x2 = x0 + right

            # --- Disjoint left/right segments with centered label gap -------------
            base_fs = max(6, style.font_size_label - 2)
            label = g.label or ""
            # Measure text width in pixels
            label_w = _text_px_width(label, style.font_label, base_fs, style.dpi)
            pad = 6.0  # px padding on each side of the label
            avail = max(4.0, x2 - x1)  # guard
            # If the text is too wide, shrink to fit up to 85% of span
            if label_w + 2 * pad > 0.85 * avail:
                scale = (0.85 * avail) / max(1.0, label_w)
                fs = max(6, int(base_fs * min(1.0, scale)))
                label_w = _text_px_width(label, style.font_label, fs, style.dpi)
            else:
                fs = base_fs
            gap = min(avail * 0.9, label_w + 2 * pad)
            xm = (x1 + x2) / 2.0
            left_end = xm - gap / 2.0
            right_start = xm + gap / 2.0

            # Draw left and right segments
            ax.plot(
                [x1, left_end],
                [y_box_center, y_box_center],
                color="#9CA3AF",
                lw=1.1,
                zorder=5,
            )
            ax.plot(
                [right_start, x2],
                [y_box_center, y_box_center],
                color="#9CA3AF",
                lw=1.1,
                zorder=5,
            )

            # Small symmetric end ticks (visual brackets)
            tick = 6.0
            ax.plot(
                [x1, x1],
                [y_box_center - tick / 2, y_box_center + tick / 2],
                color="#9CA3AF",
                lw=1.1,
                zorder=5,
            )
            ax.plot(
                [x2, x2],
                [y_box_center - tick / 2, y_box_center + tick / 2],
                color="#9CA3AF",
                lw=1.1,
                zorder=5,
            )

            if label:
                # Put the text in the gap, vertically centered on the line.
                ax.text(
                    xm,
                    y_box_center,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fs,
                    family=style.font_label,
                    color="#9CA3AF",
                    zorder=6,
                    clip_on=False,
                )
        else:
            # Unknown guide kinds are ignored intentionally (extensible, no fallback drawing)
            continue


def _draw_connectors(ax, n: int, x0: float, cw: float, y_top: float, y_bottom: float, style: Style) -> None:
    if not style.connectors or y_top <= y_bottom:
        return
    y1 = y_bottom + 0.35 * (y_top - y_bottom)
    y2 = y_top - 0.35 * (y_top - y_bottom)
    for i in range(n):
        x = x0 + i * cw + cw / 2.0
        (ln,) = ax.plot(
            [x, x],
            [y1, y2],
            color=style.color_ticks,
            lw=style.connector_width,
            alpha=style.connector_alpha,
            zorder=1,
        )
        ln.set_dashes(style.connector_dash)


def _draw_legend_centered(
    ax,
    legend: Sequence[tuple[str, str]],
    palette: Palette,
    style: Style,
    total_width: float,
) -> None:
    """One-row legend at bottom; compute real text widths and center as a block."""
    if not legend:
        return

    # Measure widths precisely
    entries: list[tuple[str, str, float]] = []
    text_total = 0.0
    for tag, label in legend:
        w_px = _text_px_width(label, style.font_label, style.legend_font_size, style.dpi)
        entries.append((tag, label, w_px))
        text_total += w_px

    n = len(entries)
    total = n * style.legend_patch_w + n * style.legend_gap_patch_text + text_total + (n - 1) * style.legend_gap_x

    x = (total_width - total) / 2.0 if style.legend_center else style.padding_x
    if x < style.padding_x:
        x = style.padding_x  # clamp

    y = style.legend_pad_px
    for i, (tag, label, w_px) in enumerate(entries):
        color = palette.color_for(tag)
        # color patch
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                style.legend_patch_w,
                style.legend_patch_h,
                boxstyle="round,pad=0.0,rounding_size=2.5",
                linewidth=0.0,
                facecolor=color,
                alpha=1.0,
                edgecolor=color,
                zorder=10,
                clip_on=False,
            )
        )
        # label
        ax.text(
            x + style.legend_patch_w + style.legend_gap_patch_text,
            y + style.legend_patch_h / 2.0,
            label,
            va="center",
            ha="left",
            fontsize=style.legend_font_size,
            family=style.font_label,
            color=style.color_sequence,
            zorder=10,
            clip_on=False,
        )
        # advance x for next item
        x += style.legend_patch_w + style.legend_gap_patch_text + w_px
        if i < n - 1:
            x += style.legend_gap_x


def _draw_overlay_label(ax, layout: LayoutResult, style: Style, text: str) -> None:
    """
    Lightweight, non-intrusive label at the top-left corner (figure space),
    e.g., 'row=42' or 'sel_row=7 id=...'. Does not affect layout or padding.
    """
    if not text:
        return
    # Small inset from the top edge
    y = layout.height - 6.0
    x = style.padding_x
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="top",
        fontsize=style.font_size_label,
        family=style.font_label,
        color="#6B7280",
        alpha=0.95,
        zorder=15,
        clip_on=False,
    )


def render_figure(
    record: SeqRecord,
    *,
    style: Style,
    palette: Palette,
    out_path: Optional[str] = None,
    fmt: str = "png",
    fixed_tracks: tuple[int, int] | None = None,
    fixed_n: Optional[int] = None,
    legend_entries: Optional[Sequence[tuple[str, str]]] = None,
):
    record = record.validate()
    layout = _compute_layout(record, style, fixed_tracks=fixed_tracks, fixed_n=fixed_n)
    fig = plt.figure(figsize=(layout.width / style.dpi, layout.height / style.dpi), dpi=style.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Center horizontally within the global column count
    n_this = len(record.sequence)
    extra_cols = max(0, (fixed_n or n_this) - n_this)
    x0 = layout.x_left + (extra_cols * layout.cw) / 2.0

    # Baselines
    _draw_sequence(ax, record.sequence, x0, layout.y_forward, layout.cw, style, "5'", "3'")
    if style.show_reverse_complement and record.alphabet == "DNA":
        _draw_sequence(
            ax,
            comp(record.sequence),
            x0,
            layout.y_reverse,
            layout.cw,
            style,
            "3'",
            "5'",
        )

    # Connector dashes (one row)
    if style.show_reverse_complement:
        _draw_connectors(ax, n_this, x0, layout.cw, layout.y_forward, layout.y_reverse, style)

    # Annotations
    up = [a for a in record.annotations if a.strand == "fwd"]
    dn = [a for a in record.annotations if a.strand == "rev"]
    for a, tr in zip(up, layout.up_tracks):
        xx = x0 + a.start * layout.cw
        yy = layout.y_forward + (tr + 1) * style.track_spacing
        color = palette.color_for(a.tag)
        _draw_box(
            ax,
            xx,
            yy,
            a.length * layout.cw,
            layout.ch * style.kmer.height_factor,
            a.label,
            color,
            style,
            above=True,
            cw=layout.cw,
        )
    for a, tr in zip(dn, layout.dn_tracks):
        xx = x0 + a.start * layout.cw
        yy = layout.y_reverse - (tr + 1) * style.track_spacing
        color = palette.color_for(a.tag)
        label_txt = a.label[::-1]
        _draw_box(
            ax,
            xx,
            yy,
            a.length * layout.cw,
            layout.ch * style.kmer.height_factor,
            label_txt,
            color,
            style,
            above=False,
            cw=layout.cw,
        )

    # Guides
    _draw_guides(
        ax,
        record.guides,
        x0,
        layout.cw,
        layout.ch,
        layout.y_forward,
        layout.n_up_tracks,
        style,
    )

    # Fixed canvas prevents squeezing and ensures nothing clips
    ax.set_xlim(0, layout.width)
    ax.set_ylim(0, layout.height)

    # Legend
    if style.legend and legend_entries:
        _draw_legend_centered(ax, legend_entries, palette, style, layout.width)

    # Optional overlay label from guides (kind='overlay_label')
    try:
        overlay = next(
            (g.label for g in record.guides if getattr(g, "kind", "") == "overlay_label" and g.label),
            None,
        )
        if overlay:
            _draw_overlay_label(ax, layout, style, overlay)
    except Exception:
        pass

    if out_path:
        fig.savefig(out_path, format=fmt, bbox_inches=None, pad_inches=0.0)
    return fig
