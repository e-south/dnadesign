"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_nucleotide_grid.py

Keep feature nucleotides on the sequence row's glyph grid.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox

from dnadesign.baserender import (
    ContractError,
    Feature,
    Palette,
    Record,
    SchemaError,
    Span,
    Style,
    initialize_runtime,
    render_record,
)


@pytest.mark.parametrize("strand", ["fwd", "rev"])
@pytest.mark.parametrize("family", ["Arial", "DejaVu Sans Mono"])
def test_feature_glyphs_match_their_sequence_columns_and_scale(strand, family):
    sequence = "ACGTTGCAAGTCCTGATCGTACCGATGCTTAGGACGT"
    complement = sequence.translate(str.maketrans("ACGT", "TGCA"))
    row = sequence if strand == "fwd" else complement
    label = row[7:23] if strand == "fwd" else row[7:23][::-1]
    initialize_runtime()
    figure = render_record(
        Record(
            id="aligned-overlap",
            alphabet="DNA",
            sequence=sequence,
            features=(Feature(id="motif", kind="kmer", span=Span(7, 23, strand), label=label),),
        ),
        renderer_name="sequence_rows",
        style=Style(font_mono=family, font_size_seq=20, font_size_feature_label=20),
        palette=Palette({"kmer": "#267C73"}),
    )
    try:
        figure.canvas.draw()
        axis = figure.axes[0]
        features = [artist for artist in axis.patches if artist.get_zorder() == 4]
        assert len(features) == 16
        glyphs = {artist.get_gid(): artist for artist in axis.patches if artist.get_gid()}
        vertical_offsets = []
        for coordinate, feature in zip(range(7, 23), features, strict=True):
            sequence_glyph = glyphs[f"sequence:{strand}:{coordinate}:{row[coordinate]}"]
            actual = feature.get_window_extent()
            expected = sequence_glyph.get_window_extent()
            np.testing.assert_allclose(
                [actual.x0, actual.x1, actual.height], [expected.x0, expected.x1, expected.height]
            )
            vertical_offsets.append(actual.y0 - expected.y0)
        np.testing.assert_allclose(vertical_offsets, vertical_offsets[0])
    finally:
        plt.close(figure)


@pytest.mark.parametrize("family", ["Arial", "DejaVu Sans Mono"])
@pytest.mark.parametrize("label_size", [10, 20, 30])
def test_feature_glyphs_are_centered_in_each_cell_at_equal_and_resized_fonts(family, label_size):
    from dnadesign.baserender.src.render.sequence_rows import _draw_feature_box

    style = Style(font_mono=family, font_size_seq=20, font_size_feature_label=label_size)
    figure, axis = plt.subplots()
    axis.set_xlim(0, 200)
    axis.set_ylim(0, 100)
    try:
        _draw_feature_box(axis, 20, 50, 160, 40, "AGTC", "#267C73", style, cw=40, ch=40)
        figure.canvas.draw()
        features = [artist for artist in axis.patches if artist.get_zorder() == 4]
        assert len(features) == 4
        for index, glyph in enumerate(features):
            bounds = glyph.get_window_extent()
            expected_center = axis.transData.transform((20 + (index + 0.5) * 40, 50))[0]
            assert (bounds.x0 + bounds.x1) / 2 == pytest.approx(expected_center)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("family", ["Arial", "DejaVu Sans Mono"])
@pytest.mark.parametrize("label_size", [10, 20, 30])
def test_feature_rectangle_and_visible_ink_share_both_centers(family, label_size):
    from dnadesign.baserender.src.render.sequence_rows import _draw_feature_box

    style = Style(font_mono=family, font_size_seq=20, font_size_feature_label=label_size)
    figure, axis = plt.subplots()
    axis.set_xlim(0, 200)
    axis.set_ylim(0, 100)
    try:
        _draw_feature_box(axis, 20, 50, 160, 100, "AGTC", "#267C73", style, cw=40, ch=40)
        figure.canvas.draw()
        rectangle = next(p for p in axis.patches if isinstance(p, FancyBboxPatch))
        ink = Bbox.union([p.get_window_extent() for p in axis.patches if p.get_zorder() == 4])
        box = rectangle.get_window_extent()
        assert (ink.x0 + ink.x1) / 2 == pytest.approx((box.x0 + box.x1) / 2)
        assert (ink.y0 + ink.y1) / 2 == pytest.approx((box.y0 + box.y1) / 2)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("family", ["Arial", "DejaVu Sans Mono"])
@pytest.mark.parametrize("strand", ["fwd", "rev"])
def test_large_nucleotide_labels_fit_without_shrinking_or_overlap(family, strand):
    sequence = "ACGTTGCAAGTCCTGATCGTACCGATGCTTAGGACGT"
    row = sequence if strand == "fwd" else sequence.translate(str.maketrans("ACGT", "TGCA"))
    label = row[7:23] if strand == "fwd" else row[7:23][::-1]
    initialize_runtime()
    figure = render_record(
        Record(
            id="large-label",
            alphabet="DNA",
            sequence=sequence,
            features=(Feature("motif", "kmer", Span(7, 23, strand), label),),
        ),
        renderer_name="sequence_rows",
        style=Style(font_mono=family, font_size_seq=12, font_size_feature_label=18, legend=False),
        palette=Palette({"kmer": "#267C73"}),
    )
    try:
        figure.canvas.draw()
        axis = figure.axes[0]
        rectangle = next(p for p in axis.patches if isinstance(p, FancyBboxPatch) and p.get_zorder() == 3)
        box = rectangle.get_window_extent()
        glyphs = [p for p in axis.patches if p.get_zorder() == 4]
        bounds = [p.get_window_extent() for p in glyphs]
        ink = Bbox.union(bounds)
        assert ink.x0 >= box.x0 and ink.x1 <= box.x1
        assert ink.y0 >= box.y0 and ink.y1 <= box.y1
        assert (ink.x0 + ink.x1) / 2 == pytest.approx((box.x0 + box.x1) / 2)
        assert (ink.y0 + ink.y1) / 2 == pytest.approx((box.y0 + box.y1) / 2)
        for first, second in zip(bounds, bounds[1:]):
            assert second.x0 - first.x1 >= 1 - 1e-6
        sequence_glyphs = {p.get_gid(): p for p in axis.patches if p.get_gid()}
        for coordinate, glyph in zip(range(7, 23), bounds, strict=True):
            source = sequence_glyphs[f"sequence:{strand}:{coordinate}:{row[coordinate]}"]
            original = source.get_window_extent()
            assert glyph.height == pytest.approx(original.height * 1.5)
            assert (glyph.x0 + glyph.x1) / 2 == pytest.approx((original.x0 + original.x1) / 2)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("field", ["pad_y_px", "min_glyph_gap_px"])
@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf"), "1"])
def test_nucleotide_geometry_rejects_invalid_padding_and_gap(field, value):
    with pytest.raises(SchemaError, match=field):
        Style(kmer={field: value})


def test_explicit_pitch_cannot_silently_overlap_larger_nucleotide_labels():
    initialize_runtime()
    before = plt.get_fignums()
    record = Record(
        id="too-tight",
        alphabet="DNA",
        sequence="ACGTACGT",
        features=(Feature("part", "kmer", Span(0, 8, "fwd"), "ACGTACGT"),),
        meta={"cell_width_scale": 0.2},
    )
    with pytest.raises(ContractError, match="pitch"):
        render_record(
            record,
            renderer_name="sequence_rows",
            style=Style(font_mono="Arial", font_size_seq=12, font_size_feature_label=18),
            palette=Palette({"kmer": "#267C73"}),
        )
    assert plt.get_fignums() == before


@pytest.mark.parametrize("family", ["Arial", "DejaVu Sans Mono"])
@pytest.mark.parametrize("alphabet,letters", [("DNA", "AGTNag"), ("IUPAC_DNA", "AGTNWM")])
def test_letter_changes_do_not_change_grid_or_rectangle_size(family, alphabet, letters):
    from dnadesign.baserender import compute_layout

    style = Style(font_mono=family, font_size_seq=12, font_size_feature_label=18, legend=False)
    layouts = []
    for letter in letters:
        record = Record(
            id="variant",
            alphabet=alphabet,
            sequence=letter * 30,
            features=(Feature("site", "kmer", Span(4, 12, "fwd"), letter * 8),),
        )
        layout = compute_layout(record, style)
        layouts.append(
            (layout.cw, layout.kmer_box_height, layout.width, layout.height, layout.y_forward, layout.y_reverse)
        )
    for layout in layouts[1:]:
        assert layout == pytest.approx(layouts[0])
