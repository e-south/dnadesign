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
from dnadesign.baserender.src.render import Palette, legend_entries_for_record, render_record
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
            "overlay_text": "details",
        },
        policies={},
        alphabet="DNA",
    )

    record = adapter.apply(row, row_index=0)
    assert record.id == "row1"
    assert len(record.features) == 2
    assert record.display.tag_labels.get("tf:lexA") == "lexA"

    style = resolve_style(preset=None, overrides={})
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    assert fig is not None
    plt.close(fig)


def test_densegen_adapter_treats_zero_offset_as_explicit_coordinate() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAAA",
        "densegen__used_tfbs_detail": [
            {"tf": "lexA", "orientation": "fwd", "tfbs": "AAA", "offset": 0},
        ],
    }
    adapter = DensegenTfbsAdapter(
        columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
        },
        policies={},
        alphabet="DNA",
    )

    record = adapter.apply(row, row_index=0)
    assert len(record.features) == 1
    assert record.features[0].span.start == 0


def test_densegen_adapter_preserves_tf_case_in_tags_and_attrs() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAAA",
        "densegen__used_tfbs_detail": [
            {"tf": "LexA", "orientation": "fwd", "tfbs": "AAA", "offset": 0},
        ],
    }
    adapter = DensegenTfbsAdapter(
        columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
        },
        policies={},
        alphabet="DNA",
    )

    record = adapter.apply(row, row_index=0)
    assert len(record.features) == 1
    assert record.features[0].tags == ("tf:LexA",)
    assert record.features[0].attrs.get("tf") == "LexA"
    assert record.display.tag_labels.get("tf:LexA") == "LexA"


def test_densegen_adapter_title_cases_background_legend_label() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAAA",
        "densegen__used_tfbs_detail": [
            {"tf": "background", "orientation": "fwd", "tfbs": "AAA", "offset": 0},
        ],
    }
    adapter = DensegenTfbsAdapter(
        columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
        },
        policies={},
        alphabet="DNA",
    )

    record = adapter.apply(row, row_index=0)
    assert record.display.tag_labels.get("tf:background") == "Background"


def test_densegen_adapter_includes_promoter_components_in_features_and_legend() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAATTTGGGCCCCCCAAAATTTGGG",
        "densegen__used_tfbs_detail": [
            {"tf": "lexA", "orientation": "fwd", "tfbs": "AAA", "offset": 0},
        ],
        "densegen__promoter_detail": {
            "placements": [
                {
                    "name": "sigma70_core",
                    "upstream_seq": "TTTGGG",
                    "downstream_seq": "AAAATT",
                    "upstream_start": 5,
                    "downstream_start": 17,
                    "spacer_length": 7,
                }
            ]
        },
    }

    adapter = DensegenTfbsAdapter(
        columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
        },
        policies={},
        alphabet="DNA",
    )

    record = adapter.apply(row, row_index=0)
    assert len(record.features) == 3
    feature_tags = {tag for feature in record.features for tag in feature.tags}
    assert "tf:lexA" in feature_tags
    assert "promoter:sigma70_core:upstream" in feature_tags
    assert "promoter:sigma70_core:downstream" in feature_tags
    assert record.display.tag_labels["promoter:sigma70_core:upstream"] == "-35 site"
    assert record.display.tag_labels["promoter:sigma70_core:downstream"] == "-10 site"

    legend = legend_entries_for_record(record)
    legend_labels = {label for _, label in legend}
    assert "-35 site" in legend_labels
    assert "-10 site" in legend_labels
