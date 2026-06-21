"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/msa/visualization/renderers/panels.py

SVG panel renderers for generic MSA visualization sidecars.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html
from pathlib import Path

from dnadesign.aligner.msa.visualization.contracts.models import (
    AnnotationTrack,
    ExemplarRow,
    PositionQc,
    ProfileQc,
)
from dnadesign.aligner.msa.visualization.contracts.panel_spec import MsaPanelSpec
from dnadesign.aligner.msa.visualization.renderers.feature_labels import resolve_label_placement


def write_alignment_overview_svg(
    path: Path,
    *,
    qc: ProfileQc,
    records: dict[str, str],
    target_row_id: str,
    tracks: tuple[AnnotationTrack, ...],
    exemplar_rows: tuple[ExemplarRow, ...],
    panel_spec: MsaPanelSpec,
) -> None:
    """Write a selected-row whole-alignment overview in target coordinates."""

    selected_rows = exemplar_rows[: panel_spec.max_display_rows]
    if not selected_rows:
        raise ValueError("alignment overview requires at least one exemplar row")
    target_aligned = records[target_row_id]
    target_columns = _target_column_by_position(target_aligned)
    width = 1440
    left_margin = 310
    right_margin = 36
    plot_width = width - left_margin - right_margin
    rect_width = max(1.0, plot_width / max(qc.canonical_position_count, 1))
    feature_band_height = max(30, 30 * max(1, len(tracks)))
    row_height = 22
    title_height = 68
    axis_y = title_height + feature_band_height + 18
    row_start_y = axis_y + 34
    height = row_start_y + len(selected_rows) * row_height + 50
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">'
        ),
        f"<title>{html.escape(qc.profile_id)} selected-row MSA overview</title>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="18" y="25" font-family="Arial, sans-serif" font-size="16" '
            f'font-weight="700">{html.escape(qc.profile_id)} selected-row overview</text>'
        ),
        (
            '<text x="18" y="44" font-family="Arial, sans-serif" font-size="11" fill="#555">'
            f"{qc.record_count} records scored; this panel shows {len(selected_rows)} explicit display rows in "
            "target-position coordinates.</text>"
        ),
    ]
    _append_feature_bands(parts, tracks=tracks, left_margin=left_margin, rect_width=rect_width, start_y=title_height)
    _append_axis(
        parts,
        max_position=qc.canonical_position_count,
        left_margin=left_margin,
        rect_width=rect_width,
        y=axis_y,
    )

    for row_index, row in enumerate(selected_rows):
        row_y = row_start_y + row_index * row_height
        sequence = records[row.record_id]
        parts.append(
            f'<text x="18" y="{row_y + 14}" font-family="Arial, sans-serif" font-size="10" '
            f'fill="#222">{html.escape(row.label)}</text>'
        )
        parts.append(
            f'<text x="250" y="{row_y + 14}" font-family="Arial, sans-serif" font-size="9" '
            f'fill="#777">{html.escape(row.group)}</text>'
        )
        for position in range(1, qc.canonical_position_count + 1):
            column = target_columns[position]
            residue = sequence[column - 1]
            target_residue = target_aligned[column - 1]
            fill = _overview_fill(
                residue=residue,
                target_residue=target_residue,
                is_target=row.record_id == target_row_id,
            )
            x = left_margin + (position - 1) * rect_width
            parts.append(
                f'<rect x="{x:.2f}" y="{row_y}" width="{rect_width + 0.05:.2f}" height="15" '
                f'fill="{fill}"><title>{html.escape(row.label)} pos {position}: '
                f"{html.escape(residue)}</title></rect>"
            )

    legend_y = height - 22
    parts.extend(
        [
            f'<rect x="{left_margin}" y="{legend_y - 10}" width="12" height="10" fill="#d9d9d9"/>',
            f'<text x="{left_margin + 17}" y="{legend_y}" font-family="Arial, sans-serif" '
            'font-size="10" fill="#555">target match</text>',
            f'<rect x="{left_margin + 112}" y="{legend_y - 10}" width="12" height="10" fill="#2166ac"/>',
            f'<text x="{left_margin + 129}" y="{legend_y}" font-family="Arial, sans-serif" '
            'font-size="10" fill="#555">substitution</text>',
            f'<rect x="{left_margin + 224}" y="{legend_y - 10}" width="12" height="10" fill="#f4a582"/>',
            f'<text x="{left_margin + 241}" y="{legend_y}" font-family="Arial, sans-serif" '
            'font-size="10" fill="#555">gap at target position</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_consensus_histogram_svg(
    path: Path,
    *,
    qc: ProfileQc,
    positions: list[PositionQc],
    tracks: tuple[AnnotationTrack, ...],
    panel_spec: MsaPanelSpec,
) -> None:
    """Write a target-position plurality and gap-fraction histogram."""

    width = 1440
    left_margin = 96
    right_margin = 36
    plot_width = width - left_margin - right_margin
    chart_top = 78
    chart_height = 160
    track_start_y = chart_top + chart_height + 48
    height = track_start_y + max(1, len(tracks)) * 30 + 52
    rect_width = max(1.0, plot_width / max(qc.canonical_position_count, 1))
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">'
        ),
        f"<title>{html.escape(qc.profile_id)} plurality histogram</title>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="18" y="25" font-family="Arial, sans-serif" font-size="16" '
            f'font-weight="700">{html.escape(qc.profile_id)} plurality histogram</text>'
        ),
        (
            '<text x="18" y="44" font-family="Arial, sans-serif" font-size="11" fill="#555">'
            "Blue bars show non-gap plurality frequency by target position; orange overlay shows gap fraction.</text>"
        ),
        f'<line x1="{left_margin}" y1="{chart_top + chart_height}" '
        f'x2="{left_margin + plot_width}" y2="{chart_top + chart_height}" stroke="#444"/>',
        f'<line x1="{left_margin}" y1="{chart_top}" x2="{left_margin}" y2="{chart_top + chart_height}" stroke="#444"/>',
    ]
    for row in positions:
        x = left_margin + (row.canonical_position - 1) * rect_width
        plurality_height = row.plurality_frequency * chart_height
        gap_height = row.gap_fraction * chart_height
        parts.append(
            f'<rect x="{x:.2f}" y="{chart_top + chart_height - plurality_height:.2f}" '
            f'width="{rect_width:.2f}" height="{plurality_height:.2f}" fill="#2166ac" opacity="0.86">'
            f"<title>pos {row.canonical_position}: plurality {row.plurality_aa} "
            f"{row.plurality_frequency:.3f}, gap {row.gap_fraction:.3f}</title></rect>"
        )
        if gap_height > 0:
            parts.append(
                f'<rect x="{x:.2f}" y="{chart_top + chart_height - gap_height:.2f}" '
                f'width="{rect_width:.2f}" height="{gap_height:.2f}" fill="#d95f02" opacity="0.42"/>'
            )
    _append_axis(
        parts,
        max_position=qc.canonical_position_count,
        left_margin=left_margin,
        rect_width=rect_width,
        y=chart_top + chart_height + 18,
    )
    _append_feature_bands(parts, tracks=tracks, left_margin=left_margin, rect_width=rect_width, start_y=track_start_y)
    if panel_spec.display_gap_trim_threshold is not None:
        parts.append(
            f'<text x="{left_margin}" y="{height - 18}" font-family="Arial, sans-serif" font-size="10" fill="#555">'
            f"display-only high-gap trim threshold declared: {panel_spec.display_gap_trim_threshold:.2f}; "
            "this target-position histogram is not trimmed.</text>"
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _append_feature_bands(
    parts: list[str],
    *,
    tracks: tuple[AnnotationTrack, ...],
    left_margin: int,
    rect_width: float,
    start_y: int,
) -> None:
    for track_index, track in enumerate(tracks):
        y = start_y + track_index * 30
        parts.append(
            f'<text x="18" y="{y + 14}" font-family="Arial, sans-serif" font-size="10" '
            f'fill="#555">{html.escape(track.label)}</text>'
        )
        for feature in track.features:
            x = left_margin + (feature.start - 1) * rect_width
            width = max(1.0, (feature.end - feature.start + 1) * rect_width)
            color = html.escape(feature.color or track.color)
            stroke = html.escape(feature.stroke_color)
            parts.append(
                f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="17" '
                f'fill="{color}" opacity="{feature.fill_opacity:.2f}" '
                f'stroke="{stroke}" stroke-width="{feature.stroke_width:.2f}">'
                f"<title>{html.escape(feature.label)} "
                f"{feature.start}-{feature.end}</title></rect>"
            )
            placement = resolve_label_placement(
                feature=feature,
                x=x,
                width=width,
                y=y,
                inside_y=y + 13,
                above_y=y - 4,
                below_y=y + 28,
                min_inside_width=26,
            )
            if placement.visible:
                parts.append(
                    f'<text x="{placement.x:.2f}" y="{placement.y:.2f}" font-family="Arial, sans-serif" '
                    f'font-size="9" text-anchor="{placement.anchor}" fill="{html.escape(placement.color)}" '
                    f'data-label-position="{html.escape(placement.position)}">'
                    f"{html.escape(feature.label)}</text>"
                )


def _append_axis(
    parts: list[str],
    *,
    max_position: int,
    left_margin: int,
    rect_width: float,
    y: int,
) -> None:
    for tick in _axis_ticks(max_position):
        x = left_margin + (tick - 1) * rect_width
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{y - 6}" x2="{x:.2f}" y2="{y + 2}" stroke="#777" stroke-width="1"/>',
                f'<text x="{x:.2f}" y="{y + 14}" font-family="Arial, sans-serif" font-size="9" '
                f'text-anchor="middle" fill="#555">{tick}</text>',
            ]
        )


def _overview_fill(*, residue: str, target_residue: str, is_target: bool) -> str:
    if residue == "-":
        return "#f4a582"
    if is_target:
        return "#a6a6a6"
    if residue == target_residue:
        return "#d9d9d9"
    if residue in {"D", "E", "K", "R", "H"}:
        return "#2166ac"
    if residue in {"F", "W", "Y", "A", "V", "I", "L", "M", "P", "G", "C"}:
        return "#4393c3"
    return "#92c5de"


def _target_column_by_position(target_aligned: str) -> dict[int, int]:
    columns: dict[int, int] = {}
    canonical_position = 0
    for column_index, residue in enumerate(target_aligned, start=1):
        if residue == "-":
            continue
        canonical_position += 1
        columns[canonical_position] = column_index
    return columns


def _axis_ticks(canonical_position_count: int) -> tuple[int, ...]:
    if canonical_position_count <= 0:
        return ()
    ticks = {1, canonical_position_count}
    step = 25 if canonical_position_count <= 160 else 50
    ticks.update(range(step, canonical_position_count + 1, step))
    return tuple(sorted(ticks))
