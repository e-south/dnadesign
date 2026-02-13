"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_motif_logo_effect.py

Motif-logo effect rendering smoke test.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from dnadesign.baserender.src.config import resolve_style
from dnadesign.baserender.src.core import Record, Span
from dnadesign.baserender.src.core.record import Display, Effect, Feature
from dnadesign.baserender.src.render import Palette, render_record
from dnadesign.baserender.src.render.effects.motif_logo import reverse_complement_matrix
from dnadesign.baserender.src.runtime import initialize_runtime


def test_motif_logo_effect_renders_for_kmer_feature() -> None:
    record = Record(
        id="r1",
        alphabet="DNA",
        sequence="ACGTACGT",
        features=(
            Feature(
                id="k1",
                kind="kmer",
                span=Span(start=0, end=4, strand="fwd"),
                label="ACGT",
                tags=("tf:lexa",),
                attrs={},
                render={},
            ),
        ),
        effects=(
            Effect(
                kind="motif_logo",
                target={"feature_id": "k1"},
                params={
                    "matrix": [
                        [0.9, 0.03, 0.04, 0.03],
                        [0.03, 0.9, 0.04, 0.03],
                        [0.03, 0.04, 0.9, 0.03],
                        [0.03, 0.04, 0.03, 0.9],
                    ]
                },
                render={},
            ),
        ),
        display=Display(),
        meta={},
    )

    style = resolve_style(preset=None, overrides={})
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    assert fig is not None
    plt.close(fig)


def test_reverse_complement_matrix_flips_columns_and_rows() -> None:
    matrix = [
        [1.0, 2.0, 3.0, 4.0],  # A C G T
        [10.0, 20.0, 30.0, 40.0],
    ]
    rc = reverse_complement_matrix(matrix)
    assert rc == [
        [40.0, 30.0, 20.0, 10.0],  # T G C A -> A C G T order after complement map
        [4.0, 3.0, 2.0, 1.0],
    ]
