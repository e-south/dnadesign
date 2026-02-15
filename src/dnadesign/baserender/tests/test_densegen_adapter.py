"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_densegen_adapter.py

DenseGen adapter contract parity smoke test with rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from dnadesign.baserender.src.adapters.densegen_tfbs import DensegenTfbsAdapter
from dnadesign.baserender.src.config import resolve_style
from dnadesign.baserender.src.render import Palette, render_record
from dnadesign.baserender.src.runtime import initialize_runtime


def test_densegen_adapter_yields_valid_record_and_renderer_works(tmp_path) -> None:
    row = {
        "id": "row1",
        "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
        "densegen__used_tfbs_detail": [
            {"tf": "lexA", "orientation": "fwd", "tfbs": "TTGACA", "offset": 0},
            {"tf": "cpxR", "orientation": "fwd", "tfbs": "TATAAT", "offset": 23},
        ],
        "details": "demo row",
    }

    adapter = DensegenTfbsAdapter(
        columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
            "details": "details",
        },
        policies={},
        alphabet="DNA",
    )

    record = adapter.apply(row, row_index=0)
    assert record.id == "row1"
    assert len(record.features) == 2
    assert record.display.tag_labels.get("tf:lexa") == "lexa"

    style = resolve_style(preset=None, overrides={})
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    assert fig is not None
    plt.close(fig)
