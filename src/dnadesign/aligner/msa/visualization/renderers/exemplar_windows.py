"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/msa/visualization/renderers/exemplar_windows.py

Selected-row motif-window SVG renderer for MSA visualization sidecars.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html

from dnadesign.aligner.msa.visualization.contracts.models import (
    AnnotationTrack,
    ExemplarRow,
    FeatureWindow,
    ProfileQc,
)


def write_exemplar_windows_svg(
    *,
    qc: ProfileQc,
    records: dict[str, str],
    target_row_id: str,
    tracks: tuple[AnnotationTrack, ...],
    exemplar_rows: tuple[ExemplarRow, ...],
) -> None:
    """Write selected rows around annotated target-position windows."""

    target_column_by_position = _target_column_by_position(records[target_row_id])
    windows = _feature_windows(tracks, flank=6, max_position=qc.canonical_position_count)
    cell_width = 14
    cell_height = 18
    label_width = 270
    window_gap = 24
    title_height = 50
    row_block_height = len(exemplar_rows) * cell_height + 42
    window_widths = [label_width + (window.end - window.start + 1) * cell_width for window in windows]
    width = max(760, min(1800, max(window_widths, default=760) + 48))
    height = title_height + len(windows) * (row_block_height + window_gap) + 20
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">'
        ),
        f"<title>{html.escape(qc.profile_id)} exemplar MSA windows</title>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="16" y="24" font-family="Arial, sans-serif" font-size="16" '
            f'font-weight="700">{html.escape(qc.profile_id)} exemplar windows</text>'
        ),
        (
            '<text x="16" y="43" font-family="Arial, sans-serif" font-size="11" fill="#555">'
            "Selected rows around annotated target-position features; dots match the target residue.</text>"
        ),
    ]
    y = title_height
    for window in windows:
        feature = window.feature
        start = window.start
        end = window.end
        parts.append(
            f'<text x="16" y="{y + 14}" font-family="Arial, sans-serif" '
            f'font-size="12" font-weight="700">{html.escape(feature.label)} ({start}-{end})</text>'
        )
        header_y = y + 30
        for offset, position in enumerate(range(start, end + 1)):
            x = label_width + offset * cell_width
            if position == start or position == end or position % 5 == 0:
                parts.append(
                    f'<text x="{x + cell_width / 2:.1f}" y="{header_y}" '
                    'font-family="Arial, sans-serif" font-size="8" text-anchor="middle" '
                    f'fill="#555">{position}</text>'
                )
        row_y = y + 36
        feature_x = label_width + (feature.start - start) * cell_width
        feature_width = (feature.end - feature.start + 1) * cell_width
        feature_height = len(exemplar_rows) * cell_height
        parts.append(
            f'<rect x="{feature_x:.1f}" y="{row_y}" width="{feature_width:.1f}" height="{feature_height}" '
            f'fill="{html.escape(feature.color)}" opacity="{feature.fill_opacity:.2f}" '
            f'stroke="{html.escape(feature.stroke_color)}" stroke-width="{feature.stroke_width:.2f}" '
            f'fill-opacity="{feature.fill_opacity:.2f}" data-feature-id="{html.escape(feature.id)}">'
            f"<title>{html.escape(feature.label)} exact span {feature.start}-{feature.end}</title></rect>"
        )
        for row_index, exemplar in enumerate(exemplar_rows):
            current_y = row_y + row_index * cell_height
            parts.append(
                f'<text x="16" y="{current_y + 13}" font-family="Arial, sans-serif" '
                f'font-size="10" fill="#333">{html.escape(exemplar.label)}</text>'
            )
            sequence = records[exemplar.record_id]
            for offset, position in enumerate(range(start, end + 1)):
                column = target_column_by_position[position]
                residue = sequence[column - 1]
                target_residue = records[target_row_id][column - 1]
                display = "." if exemplar.record_id != target_row_id and residue == target_residue else residue
                x = label_width + offset * cell_width
                fill = _residue_fill(residue=residue, target_residue=target_residue)
                parts.append(
                    f'<rect x="{x:.1f}" y="{current_y}" width="{cell_width}" height="{cell_height}" '
                    f'fill="{fill}" stroke="#ffffff" stroke-width="0.5">'
                    f"<title>{html.escape(exemplar.label)} pos {position}: {html.escape(residue)}</title></rect>"
                )
                parts.append(
                    f'<text x="{x + cell_width / 2:.1f}" y="{current_y + 13}" '
                    'font-family="Menlo, Consolas, monospace" font-size="10" '
                    f'text-anchor="middle" fill="#222">{html.escape(display)}</text>'
                )
        parts.append(
            f'<rect x="{feature_x:.1f}" y="{row_y}" width="{feature_width:.1f}" height="{feature_height}" '
            f'fill="none" stroke="{html.escape(feature.stroke_color)}" stroke-width="{feature.stroke_width:.2f}" '
            f'data-feature-id="{html.escape(feature.id)}-outline">'
            f"<title>{html.escape(feature.label)} exact span outline {feature.start}-{feature.end}</title></rect>"
        )
        y += row_block_height + window_gap
    parts.append("</svg>")
    qc.profile_exemplar_svg_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _target_column_by_position(target_aligned: str) -> dict[int, int]:
    columns: dict[int, int] = {}
    canonical_position = 0
    for column_index, residue in enumerate(target_aligned, start=1):
        if residue == "-":
            continue
        canonical_position += 1
        columns[canonical_position] = column_index
    return columns


def _feature_windows(
    tracks: tuple[AnnotationTrack, ...],
    *,
    flank: int,
    max_position: int,
) -> tuple[FeatureWindow, ...]:
    windows: list[FeatureWindow] = []
    for track in tracks:
        for feature in track.features:
            windows.append(
                FeatureWindow(
                    feature=feature,
                    start=max(1, feature.start - flank),
                    end=min(max_position, feature.end + flank),
                )
            )
    return tuple(windows)


def _residue_fill(*, residue: str, target_residue: str) -> str:
    if residue == "-":
        return "#f2c6c2"
    if residue == target_residue:
        return "#f3f5f7"
    if residue in {"D", "E"}:
        return "#c7dcef"
    if residue in {"K", "R", "H"}:
        return "#f6d7a7"
    if residue in {"S", "T", "N", "Q"}:
        return "#d8ead2"
    if residue in {"F", "W", "Y"}:
        return "#e5d6ef"
    if residue in {"A", "V", "I", "L", "M", "P", "G", "C"}:
        return "#e8e1d5"
    return "#eeeeee"
