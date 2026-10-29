"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_sequence_rows_visual_metadata.py

Tests for sequence-row visual metadata overlays.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from dnadesign.baserender.src.config import resolve_style
from dnadesign.baserender.src.core import Display, Feature, Record, Span
from dnadesign.baserender.src.render import Palette, render_record
from dnadesign.baserender.src.runtime import initialize_runtime


def test_sequence_rows_draws_explicit_boundary_ticks_and_span_bracket() -> None:
    initialize_runtime()
    record = Record(
        id="boundary_ticks",
        alphabet="DNA",
        sequence="TCCCTATCAGTGATAGAGA",
        features=(
            Feature(
                id="retained",
                kind="kmer",
                span=Span(start=2, end=17, strand="fwd"),
                label="CCTATCAGTGATAGA",
                tags=("tf:tetR_trim",),
                attrs={"style_token": "tf:tetR_trim"},
            ),
        ),
        effects=(),
        display=Display(tag_labels={"tf:tetR_trim": "retained tetO payload"}),
        meta={
            "boundary_ticks": [
                {"position": 2, "label": "2", "emphasis": "active"},
                {"position": 17, "label": "17", "emphasis": "active"},
            ],
            "span_brackets": [
                {
                    "target_feature_id": "retained",
                    "label": "retained payload",
                    "color": "#3E927F",
                }
            ],
        },
    ).validate()
    style = resolve_style(
        preset="presentation_default",
        overrides={
            "show_reverse_complement": True,
            "show_coordinate_ticks": True,
            "palette": {"tf:tetR_trim": "#3E927F"},
        },
    )
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=Palette(style.palette))
    try:
        ax = fig.axes[0]
        text_by_gid = {artist.get_gid(): artist.get_text() for artist in ax.texts if artist.get_gid()}
        line_gids = {line.get_gid() for line in ax.lines if line.get_gid()}

        assert text_by_gid["sequence_boundary_tick_label:0:2"] == "2"
        assert text_by_gid["sequence_boundary_tick_label:1:17"] == "17"
        assert "sequence_boundary_tick:0:2" in line_gids
        assert "sequence_boundary_tick:1:17" in line_gids
        assert text_by_gid["sequence_span_bracket_label:0:retained"] == "retained payload"
        assert "sequence_span_bracket:0:retained" in line_gids
    finally:
        plt.close(fig)
