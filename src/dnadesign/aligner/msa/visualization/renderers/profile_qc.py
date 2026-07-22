"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/msa/visualization/renderers/profile_qc.py

Target-position QC SVG renderer for MSA visualization sidecars.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html

from dnadesign.aligner.msa.visualization.contracts.models import (
    AnnotationTrack,
    PositionQc,
    ProfileQc,
)
from dnadesign.aligner.msa.visualization.renderers.feature_labels import resolve_label_placement


def write_profile_qc_svg(qc: ProfileQc, positions: list[PositionQc], tracks: tuple[AnnotationTrack, ...]) -> None:
    """Write a target-position QC SVG with optional annotation tracks."""

    width = max(720, min(1280, qc.canonical_position_count * 4 + 160))
    left_margin = 96
    right_margin = 24
    plot_width = width - left_margin - right_margin
    rect_width = max(1.0, plot_width / max(qc.canonical_position_count, 1))
    track_row_height = 30
    track_start_y = 122
    height = track_start_y + max(1, len(tracks)) * track_row_height + 42
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">'
        ),
        f"<title>{html.escape(qc.profile_id)} MSA position QC</title>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="16" y="24" font-family="Arial, sans-serif" font-size="16" '
            f'font-weight="700">{html.escape(qc.profile_id)}</text>'
        ),
        (
            '<text x="16" y="43" font-family="Arial, sans-serif" font-size="11" '
            f'fill="#555">{qc.record_count} records, {qc.canonical_position_count} target positions, '
            f"{qc.inserted_column_count} target-gap columns</text>"
        ),
    ]
    for index, row in enumerate(positions):
        x = left_margin + index * rect_width
        hue = _gap_color(row.gap_fraction)
        parts.append(
            f'<rect x="{x:.2f}" y="56" width="{rect_width:.2f}" height="28" '
            f'fill="{hue}"><title>pos {row.canonical_position}: '
            f"gap {row.gap_fraction:.3f}</title></rect>"
        )
    parts.append(
        f'<line x1="{left_margin}" y1="94" x2="{left_margin + plot_width}" y2="94" stroke="#777" stroke-width="1"/>'
    )
    for tick in _axis_ticks(qc.canonical_position_count):
        x = left_margin + (tick - 1) * rect_width
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="90" x2="{x:.2f}" y2="98" stroke="#777" stroke-width="1"/>',
                (
                    f'<text x="{x:.2f}" y="111" font-family="Arial, sans-serif" '
                    f'font-size="10" text-anchor="middle" fill="#555">{tick}</text>'
                ),
            ]
        )
    if tracks:
        for track_index, track in enumerate(tracks):
            y = track_start_y + track_index * track_row_height
            parts.append(
                f'<text x="16" y="{y + 15}" font-family="Arial, sans-serif" '
                f'font-size="11" fill="#333">{html.escape(track.label)}</text>'
            )
            for feature in track.features:
                x = left_margin + (feature.start - 1) * rect_width
                feature_width = max(1.0, (feature.end - feature.start + 1) * rect_width)
                color = html.escape(feature.color or track.color)
                stroke = html.escape(feature.stroke_color)
                stroke_width = feature.stroke_width
                parts.append(
                    f'<rect x="{x:.2f}" y="{y}" width="{feature_width:.2f}" height="18" '
                    f'rx="2" ry="2" fill="{color}" opacity="{feature.fill_opacity:.2f}" '
                    f'stroke="{stroke}" stroke-width="{stroke_width:.2f}">'
                    f"<title>{html.escape(feature.label)}: {feature.start}-{feature.end}</title></rect>"
                )
                placement = resolve_label_placement(
                    feature=feature,
                    x=x,
                    width=feature_width,
                    y=y,
                    inside_y=y + 13,
                    above_y=y - 4,
                    below_y=y + 31,
                    min_inside_width=28,
                )
                if placement.visible:
                    text_color = html.escape(placement.color)
                    parts.append(
                        f'<text x="{placement.x:.2f}" y="{placement.y:.2f}" '
                        'font-family="Arial, sans-serif" font-size="10" '
                        f'text-anchor="{placement.anchor}" fill="{text_color}" '
                        f'data-label-position="{html.escape(placement.position)}">'
                        f"{html.escape(feature.label)}</text>"
                    )
    else:
        y = track_start_y
        parts.append(
            f'<text x="16" y="{y + 15}" font-family="Arial, sans-serif" '
            'font-size="11" fill="#777">No annotation tracks</text>'
        )
    parts.extend(
        [
            (
                f'<text x="{left_margin}" y="{height - 16}" font-family="Arial, sans-serif" '
                'font-size="11" fill="#555">top track: lower gap fraction to higher gap fraction; '
                "lower tracks: optional target-position annotations</text>"
            ),
            "</svg>",
        ]
    )
    qc.profile_svg_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _gap_color(gap_fraction: float) -> str:
    bounded = max(0.0, min(gap_fraction, 1.0))
    red = int(230 * bounded + 40 * (1 - bounded))
    green = int(92 * bounded + 150 * (1 - bounded))
    blue = int(80 * bounded + 160 * (1 - bounded))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _axis_ticks(canonical_position_count: int) -> tuple[int, ...]:
    if canonical_position_count <= 0:
        return ()
    ticks = {1, canonical_position_count}
    ticks.update(range(50, canonical_position_count + 1, 50))
    return tuple(sorted(ticks))
