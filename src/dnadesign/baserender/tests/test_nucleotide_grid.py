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

from dnadesign.baserender import Feature, Palette, Record, Span, Style, initialize_runtime, render_record


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
