"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_motif_logo_coloring.py

Tests for motif-logo observed-sequence coloring and left-adjacent scale-bar placement.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pytest
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from dnadesign.baserender.src.config import resolve_style
from dnadesign.baserender.src.core import Record, Span
from dnadesign.baserender.src.core.record import Display, Effect, Feature
from dnadesign.baserender.src.render import Palette, render_record
from dnadesign.baserender.src.render.effects.motif_logo import compute_motif_logo_geometry
from dnadesign.baserender.src.render.layout import compute_layout
from dnadesign.baserender.src.render.sequence_rows import _sequence_tone_strengths
from dnadesign.baserender.src.runtime import initialize_runtime


def _motif_matrix(length: int) -> list[list[float]]:
    return [[0.70, 0.10, 0.10, 0.10] for _ in range(length)]


def test_reverse_strand_geometry_aligns_logo_columns_to_antisense_row() -> None:
    initialize_runtime()
    sequence = "ATACAGTT"
    segment = sequence[0:6]
    complement = segment.translate(str.maketrans("ACGT", "TGCA"))
    label = segment.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    matrix = [
        [0.90, 0.05, 0.03, 0.02],
        [0.05, 0.60, 0.30, 0.05],
        [0.20, 0.10, 0.10, 0.60],
        [0.05, 0.20, 0.70, 0.05],
        [0.65, 0.10, 0.20, 0.05],
        [0.10, 0.75, 0.10, 0.05],
    ]
    record = Record(
        id="reverse_orientation",
        alphabet="DNA",
        sequence=sequence,
        features=(
            Feature(
                id="k1",
                kind="kmer",
                span=Span(start=0, end=6, strand="rev"),
                label=label,
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="motif_logo",
                target={"feature_id": "k1"},
                params={"matrix": matrix},
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )
    style = resolve_style(
        preset=None,
        overrides={
            "motif_logo": {
                "letter_coloring": {
                    "mode": "match_window_seq",
                },
                "lane_mode": "follow_feature_track",
            }
        },
    )
    layout = compute_layout(record, style)
    geometry = compute_motif_logo_geometry(record=record, effect_index=0, layout=layout, style=style)
    assert geometry.observed == complement
    assert geometry.matrix == tuple(tuple(float(v) for v in row) for row in matrix[::-1])


def test_match_window_seq_coloring_uses_feature_fill_color() -> None:
    initialize_runtime()
    record = Record(
        id="coloring",
        alphabet="DNA",
        sequence="ACGTACGT",
        features=(
            Feature(
                id="k1",
                kind="kmer",
                span=Span(start=0, end=4, strand="fwd"),
                label="ACGT",
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="motif_logo",
                target={"feature_id": "k1"},
                params={"matrix": _motif_matrix(4)},
                render={},
            ),
        ),
        display=Display(tag_labels={"tf:lexA": "lexA"}),
        meta={},
    )
    style = resolve_style(
        preset=None,
        overrides={
            "connectors": False,
            "palette": {"tf:lexA": "#B45309"},
            "motif_logo": {
                "letter_coloring": {
                    "mode": "match_window_seq",
                    "other_color": "#d1d5db",
                    "observed_color_source": "feature_fill",
                },
                "alpha_other": 1.0,
                "alpha_observed": 1.0,
            },
        },
    )
    palette = Palette(style.palette)
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    ax = fig.axes[0]

    expected_highlight = mcolors.to_hex("#B45309")
    expected_other = mcolors.to_hex("#d1d5db")
    observed = "ACGT"

    for col_idx, observed_base in enumerate(observed):
        for base in ("A", "C", "G", "T"):
            gid = f"motif_logo:k1:{col_idx}:{base}"
            patch = next(p for p in ax.patches if p.get_gid() == gid)
            got = mcolors.to_hex(patch.get_facecolor(), keep_alpha=False)
            expected = expected_highlight if base == observed_base else expected_other
            assert got == expected

    plt.close(fig)


def test_left_of_logo_scale_bar_draws_adjacent_to_logo_bounds() -> None:
    initialize_runtime()
    record = Record(
        id="scale_bar",
        alphabet="DNA",
        sequence="ACGTACGT",
        features=(
            Feature(
                id="k1",
                kind="kmer",
                span=Span(start=1, end=5, strand="fwd"),
                label="CGTA",
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="motif_logo",
                target={"feature_id": "k1"},
                params={"matrix": _motif_matrix(4)},
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )
    style = resolve_style(
        preset=None,
        overrides={
            "connectors": False,
            "motif_logo": {
                "scale_bar": {"enabled": True, "location": "left_of_logo"},
            },
        },
    )
    layout = compute_layout(record, style)
    geometry = compute_motif_logo_geometry(record=record, effect_index=0, layout=layout, style=style)
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=Palette(style.palette))
    ax = fig.axes[0]

    vertical_lines = []
    for line in ax.lines:
        x_data = line.get_xdata(orig=False)
        y_data = line.get_ydata(orig=False)
        if len(x_data) == 2 and len(y_data) == 2 and abs(float(x_data[0]) - float(x_data[1])) < 1e-6:
            vertical_lines.append(line)

    assert vertical_lines
    x_bar = min(float(line.get_xdata(orig=False)[0]) for line in vertical_lines)
    assert x_bar < geometry.x0
    assert (geometry.x0 - x_bar) < (2.0 * layout.ch)

    plt.close(fig)


def test_left_of_logo_scale_bar_draws_one_bar_per_motif_geometry() -> None:
    initialize_runtime()
    record = Record(
        id="scale_bar_per_motif",
        alphabet="DNA",
        sequence="ACTGCATATATTTACAA",
        features=(
            Feature(
                id="k1",
                kind="regulator_window",
                span=Span(start=0, end=15, strand="fwd"),
                label="ACTGCATATATTTAC",
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
            Feature(
                id="k2",
                kind="regulator_window",
                span=Span(start=3, end=14, strand="fwd"),
                label="GCATATATTTA",
                tags=("tf:cpxR",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="motif_logo",
                target={"feature_id": "k1"},
                params={"matrix": _motif_matrix(15)},
                render={},
            ),
            Effect(
                kind="motif_logo",
                target={"feature_id": "k2"},
                params={"matrix": _motif_matrix(11)},
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )
    style = resolve_style(
        preset=None,
        overrides={
            "connectors": False,
            "motif_logo": {
                "layout": "stack",
                "lane_mode": "follow_feature_track",
                "scale_bar": {"enabled": True, "location": "left_of_logo", "pad_cells": 0.35},
            },
        },
    )
    layout = compute_layout(record, style)
    geometries = tuple(
        compute_motif_logo_geometry(record=record, effect_index=idx, layout=layout, style=style)
        for idx, _ in enumerate(record.effects)
    )

    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=Palette(style.palette))
    ax = fig.axes[0]

    vertical_segments: list[tuple[float, float, float]] = []
    for line in ax.lines:
        x_data = line.get_xdata(orig=False)
        y_data = line.get_ydata(orig=False)
        if len(x_data) != 2 or len(y_data) != 2:
            continue
        x0, x1 = float(x_data[0]), float(x_data[1])
        if abs(x0 - x1) > 1e-6:
            continue
        y0, y1 = sorted((float(y_data[0]), float(y_data[1])))
        vertical_segments.append((x0, y0, y1))

    assert vertical_segments
    pad = float(style.motif_logo.scale_bar.pad_cells) * layout.ch
    tol = 1e-2
    for geometry in geometries:
        expected_x = geometry.x0 - pad
        expected_y0 = geometry.y0
        expected_y1 = geometry.y0 + geometry.height
        assert any(
            abs(x - expected_x) <= tol and abs(y0 - expected_y0) <= tol and abs(y1 - expected_y1) <= tol
            for x, y0, y1 in vertical_segments
        ), (expected_x, expected_y0, expected_y1)

    merged_y0 = min(g.y0 for g in geometries)
    merged_y1 = max(g.y0 + g.height for g in geometries)
    merged_x0 = min(g.x0 for g in geometries)
    assert not any(
        abs(y0 - merged_y0) <= tol and abs(y1 - merged_y1) <= tol and abs((x + pad) - merged_x0) <= 1.0
        for x, y0, y1 in vertical_segments
    )

    plt.close(fig)


def test_inline_legend_labels_are_tag_colored_and_track_aligned() -> None:
    initialize_runtime()
    record = Record(
        id="inline_legend",
        alphabet="DNA",
        sequence="ACTGCATATATTTACAA",
        features=(
            Feature(
                id="k1",
                kind="regulator_window",
                span=Span(start=0, end=15, strand="fwd"),
                label="ACTGCATATATTTAC",
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
        ),
        effects=(),
        display=Display(tag_labels={"tf:lexA": "lexA"}),
        meta={},
    )
    base_overrides = {
        "connectors": False,
        "palette": {"tf:lexA": "#B45309"},
        "legend": True,
    }
    inline_style = resolve_style(
        preset=None,
        overrides={
            **base_overrides,
            "legend_mode": "inline",
            "legend_inline_side": "right",
            "legend_inline_margin_cells": 0.20,
        },
    )
    bottom_style = resolve_style(
        preset=None,
        overrides={
            **base_overrides,
            "legend_mode": "bottom",
        },
    )
    inline_layout = compute_layout(record, inline_style)
    bottom_layout = compute_layout(record, bottom_style)
    assert inline_layout.height < bottom_layout.height

    fig = render_record(
        record,
        renderer_name="sequence_rows",
        style=inline_style,
        palette=Palette(inline_style.palette),
    )
    ax = fig.axes[0]
    labels = [t for t in ax.texts if t.get_text() == "LexA"]
    assert labels, [t.get_text() for t in ax.texts]
    label = labels[0]
    assert mcolors.to_hex(label.get_color()) == mcolors.to_hex("#B45309")

    placement = inline_layout.placements[0]
    x, y = label.get_position()
    assert y == pytest.approx(placement.y)
    assert x > (placement.x + placement.w)

    plt.close(fig)


def test_inline_legend_labels_do_not_overlap_neighbor_kmer_boxes() -> None:
    initialize_runtime()
    record = Record(
        id="inline_legend_no_overlap",
        alphabet="DNA",
        sequence="ACTGCATATATTTACAACTGCA",
        features=(
            Feature(
                id="k1",
                kind="regulator_window",
                span=Span(start=0, end=6, strand="fwd"),
                label="ACTGCA",
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
            Feature(
                id="k2",
                kind="regulator_window",
                span=Span(start=7, end=13, strand="fwd"),
                label="TATATT",
                tags=("tf:cpxR",),
                attrs={},
                render={},
            ),
        ),
        effects=(),
        display=Display(tag_labels={"tf:lexA": "lexA", "tf:cpxR": "cpxR"}),
        meta={},
    )
    style = resolve_style(
        preset=None,
        overrides={
            "connectors": False,
            "palette": {"tf:lexA": "#B45309", "tf:cpxR": "#0F766E"},
            "legend": True,
            "legend_mode": "inline",
            "legend_inline_side": "right",
            "legend_inline_margin_cells": 0.05,
        },
    )
    layout = compute_layout(record, style)
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=Palette(style.palette))
    ax = fig.axes[0]

    labels = [t for t in ax.texts if t.get_text() in {"LexA", "CpxR"}]
    assert len(labels) == 2

    feature_for_label = {"LexA": "k1", "CpxR": "k2"}
    prop = FontProperties(family=style.font_label, size=style.legend_font_size)
    px_per_pt = style.dpi / 72.0
    for text_artist in labels:
        label = str(text_artist.get_text())
        own_feature_id = feature_for_label[label]
        text_width = TextPath((0, 0), label, prop=prop).get_extents().width * px_per_pt
        x_anchor, y_anchor = text_artist.get_position()
        ha = str(text_artist.get_ha()).lower()
        if ha == "left":
            text_x0 = float(x_anchor)
            text_x1 = float(x_anchor) + float(text_width)
        elif ha == "right":
            text_x0 = float(x_anchor) - float(text_width)
            text_x1 = float(x_anchor)
        else:
            half = float(text_width) * 0.5
            text_x0 = float(x_anchor) - half
            text_x1 = float(x_anchor) + half
        for feature_id, (bx0, by0, bx1, by1) in layout.feature_boxes.items():
            if feature_id == own_feature_id:
                continue
            if not (float(by0) <= float(y_anchor) <= float(by1)):
                continue
            overlaps = max(float(text_x0), float(bx0)) < min(float(text_x1), float(bx1))
            assert not overlaps, (label, (text_x0, text_x1), (bx0, bx1))

    plt.close(fig)


def test_sequence_tone_strengths_use_information_weighted_observed_probability() -> None:
    initialize_runtime()
    record = Record(
        id="tone_strengths",
        alphabet="DNA",
        sequence="AAAAA",
        features=(
            Feature(
                id="f1",
                kind="regulator_window",
                span=Span(start=0, end=5, strand="fwd"),
                label="AAAAA",
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="motif_logo",
                target={"feature_id": "f1"},
                params={
                    "matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.9, 0.1, 0.0, 0.0],
                        [0.7, 0.1, 0.1, 0.1],
                        [0.5, 0.5, 0.0, 0.0],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                },
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )
    style = resolve_style(
        preset=None,
        overrides={
            "sequence": {"bold_consensus_bases": True},
            "motif_logo": {"lane_mode": "follow_feature_track"},
        },
    )
    layout = compute_layout(record, style)
    geometries = tuple(
        compute_motif_logo_geometry(record=record, effect_index=idx, layout=layout, style=style)
        for idx, _ in enumerate(record.effects)
    )

    tone_fwd, tone_rev = _sequence_tone_strengths(record, geometries, q_low=0.10, q_high=0.90)
    assert len(tone_fwd) == len(record.sequence)
    assert len(tone_rev) == len(record.sequence)
    assert all(0.0 <= v <= 1.0 for v in tone_fwd)
    assert tone_fwd[0] == pytest.approx(1.0, abs=1e-6)
    assert tone_fwd[4] == pytest.approx(0.0, abs=1e-6)
    assert tone_fwd[1] > tone_fwd[2] > tone_fwd[3] > tone_fwd[4]


def test_sequence_tone_coloring_varies_with_strength() -> None:
    initialize_runtime()
    record = Record(
        id="tone_coloring",
        alphabet="DNA",
        sequence="AAAAA",
        features=(
            Feature(
                id="f1",
                kind="regulator_window",
                span=Span(start=0, end=5, strand="fwd"),
                label="AAAAA",
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="motif_logo",
                target={"feature_id": "f1"},
                params={
                    "matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.9, 0.1, 0.0, 0.0],
                        [0.7, 0.1, 0.1, 0.1],
                        [0.5, 0.5, 0.0, 0.0],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                },
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )
    style = resolve_style(
        preset=None,
        overrides={
            "color_sequence": "#1f2937",
            "sequence": {
                "bold_consensus_bases": True,
                "non_consensus_color": "#9ca3af",
                "tone_quantile_low": 0.10,
                "tone_quantile_high": 0.90,
            },
            "motif_logo": {"lane_mode": "follow_feature_track"},
        },
    )
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=Palette(style.palette))
    ax = fig.axes[0]

    fwd = {int(p.get_gid().split(":")[2]): p for p in ax.patches if (p.get_gid() or "").startswith("sequence:fwd:")}
    assert set(fwd.keys()) == set(range(len(record.sequence)))

    def _luma(idx: int) -> float:
        rgb = mcolors.to_rgb(fwd[idx].get_facecolor())
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    # Lower luminance = darker glyph color.
    assert _luma(0) < _luma(1) < _luma(2) < _luma(3) < _luma(4)

    plt.close(fig)


def test_sequence_tone_coloring_rev_row_stays_light_when_all_motifs_are_fwd() -> None:
    initialize_runtime()
    record = Record(
        id="tone_fwd_only",
        alphabet="DNA",
        sequence="AAAAA",
        features=(
            Feature(
                id="f1",
                kind="regulator_window",
                span=Span(start=0, end=5, strand="fwd"),
                label="AAAAA",
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="motif_logo",
                target={"feature_id": "f1"},
                params={"matrix": [[1.0, 0.0, 0.0, 0.0]] * 5},
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )
    style = resolve_style(
        preset=None,
        overrides={
            "color_sequence": "#111827",
            "sequence": {
                "bold_consensus_bases": True,
                "non_consensus_color": "#9ca3af",
                "tone_quantile_low": 0.10,
                "tone_quantile_high": 0.90,
            },
            "motif_logo": {"lane_mode": "follow_feature_track"},
        },
    )
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=Palette(style.palette))
    ax = fig.axes[0]

    dark = mcolors.to_hex("#111827")
    light = mcolors.to_hex("#9ca3af")
    fwd = {int(p.get_gid().split(":")[2]): p for p in ax.patches if (p.get_gid() or "").startswith("sequence:fwd:")}
    rev = {int(p.get_gid().split(":")[2]): p for p in ax.patches if (p.get_gid() or "").startswith("sequence:rev:")}
    assert set(fwd.keys()) == set(range(len(record.sequence)))
    assert set(rev.keys()) == set(range(len(record.sequence)))

    for idx in range(len(record.sequence)):
        fwd_hex = mcolors.to_hex(fwd[idx].get_facecolor(), keep_alpha=False)
        rev_hex = mcolors.to_hex(rev[idx].get_facecolor(), keep_alpha=False)
        assert fwd_hex == dark
        assert rev_hex == light

    plt.close(fig)


def test_sequence_tone_strengths_stacked_same_strand_do_not_dilute_on_zero_info_overlap() -> None:
    initialize_runtime()
    informative_matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.7, 0.1, 0.1, 0.1],
        [0.5, 0.5, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ]
    uniform_matrix = [[0.25, 0.25, 0.25, 0.25] for _ in range(5)]

    stacked = Record(
        id="tone_stacked",
        alphabet="DNA",
        sequence="AAAAA",
        features=(
            Feature(
                id="f_info",
                kind="regulator_window",
                span=Span(start=0, end=5, strand="fwd"),
                label="AAAAA",
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
            Feature(
                id="f_uniform",
                kind="regulator_window",
                span=Span(start=0, end=5, strand="fwd"),
                label="AAAAA",
                tags=("tf:cpxR",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="motif_logo",
                target={"feature_id": "f_info"},
                params={"matrix": informative_matrix},
                render={},
            ),
            Effect(
                kind="motif_logo",
                target={"feature_id": "f_uniform"},
                params={"matrix": uniform_matrix},
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )
    single = Record(
        id="tone_single",
        alphabet="DNA",
        sequence="AAAAA",
        features=(
            Feature(
                id="f_info",
                kind="regulator_window",
                span=Span(start=0, end=5, strand="fwd"),
                label="AAAAA",
                tags=("tf:lexA",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="motif_logo",
                target={"feature_id": "f_info"},
                params={"matrix": informative_matrix},
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )
    style = resolve_style(
        preset=None,
        overrides={
            "sequence": {"bold_consensus_bases": True},
            "motif_logo": {"lane_mode": "follow_feature_track"},
        },
    )

    stacked_layout = compute_layout(stacked, style)
    stacked_geometries = tuple(
        compute_motif_logo_geometry(record=stacked, effect_index=idx, layout=stacked_layout, style=style)
        for idx, _ in enumerate(stacked.effects)
    )
    stacked_fwd, stacked_rev = _sequence_tone_strengths(stacked, stacked_geometries, q_low=0.10, q_high=0.90)

    single_layout = compute_layout(single, style)
    single_geometries = tuple(
        compute_motif_logo_geometry(record=single, effect_index=idx, layout=single_layout, style=style)
        for idx, _ in enumerate(single.effects)
    )
    single_fwd, single_rev = _sequence_tone_strengths(single, single_geometries, q_low=0.10, q_high=0.90)

    assert stacked_fwd == pytest.approx(single_fwd)
    assert stacked_rev == pytest.approx(single_rev)
