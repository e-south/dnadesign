"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_render_semantics.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.baserender.src.contracts import AlphabetError
from dnadesign.baserender.src.model import Annotation, SeqRecord
from dnadesign.baserender.src.palette import Palette
from dnadesign.baserender.src.presets.style_presets import resolve_style
from dnadesign.baserender.src.render import render_figure


def test_render_no_connectors_for_protein_even_if_style_true():
    style = resolve_style(overrides={"show_reverse_complement": True})
    rec = SeqRecord(id="p1", alphabet="PROTEIN", sequence="ACDEFG").validate()
    pal = Palette(style.palette)
    fig = render_figure(rec, style=style, palette=pal, out_path=None)
    ax = fig.axes[0]
    # No connectors should be drawn for non-DNA records
    assert len(ax.lines) == 0
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_render_raises_on_rev_annotations_for_protein():
    style = resolve_style(overrides={"show_reverse_complement": True})
    rec = SeqRecord(
        id="p2",
        alphabet="PROTEIN",
        sequence="ACDEFG",
        annotations=(
            Annotation(
                start=0,
                length=2,
                strand="rev",
                label="AC",
                tag="tf:lexa",
            ),
        ),
    )
    pal = Palette(style.palette)
    with pytest.raises(AlphabetError):
        _ = render_figure(rec, style=style, palette=pal, out_path=None)
