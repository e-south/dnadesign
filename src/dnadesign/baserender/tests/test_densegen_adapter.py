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
import pytest
from matplotlib.patches import FancyBboxPatch, PathPatch

from dnadesign.baserender.src.adapters.densegen_tfbs import DensegenTfbsAdapter
from dnadesign.baserender.src.config import resolve_style
from dnadesign.baserender.src.core import SchemaError
from dnadesign.baserender.src.render import Palette, legend_entries_for_record, render_record
from dnadesign.baserender.src.render.layout import compute_layout
from dnadesign.baserender.src.render.sequence_rows import _span_link_label_boxes
from dnadesign.baserender.src.runtime import initialize_runtime


def test_densegen_adapter_accepts_regulator_sequence_contract() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAAA",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "AAA", "offset": 0},
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
    assert record.features[0].attrs.get("tf") == "lexA"
    assert record.features[0].label == "AAA"


def test_densegen_adapter_rejects_legacy_tf_tfbs_contract() -> None:
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

    with pytest.raises(SchemaError, match="regulator"):
        adapter.apply(row, row_index=0)


def test_densegen_adapter_yields_valid_record_and_renderer_works(tmp_path) -> None:
    row = {
        "id": "row1",
        "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
            {"regulator": "cpxR", "orientation": "fwd", "sequence": "TATAAT", "offset": 23},
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
    assert record.display.tag_labels.get("tf:lexA") == "LexA"

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
            {"regulator": "lexA", "orientation": "fwd", "sequence": "AAA", "offset": 0},
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
            {"regulator": "LexA", "orientation": "fwd", "sequence": "AAA", "offset": 0},
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
            {"regulator": "background", "orientation": "fwd", "sequence": "AAA", "offset": 0},
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


def test_densegen_adapter_title_cases_lowercase_tf_legend_labels() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAAA",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "AAA", "offset": 0},
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
    assert record.display.tag_labels.get("tf:lexA") == "LexA"


def test_densegen_adapter_includes_promoter_components_in_features_and_legend() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAATTTGGGCCCCCCAAAATTTGGG",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "AAA", "offset": 0},
        ],
        "densegen__promoter_detail": {
            "placements": [
                {
                    "name": "sigma70_core",
                    "upstream_seq": "TTTGGG",
                    "downstream_seq": "AAAATT",
                    "upstream_start": 5,
                    "downstream_start": 17,
                    "spacer_length": 6,
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
    assert record.display.tag_labels["promoter:sigma70_core:upstream"] == "σ70 -35 site"
    assert record.display.tag_labels["promoter:sigma70_core:downstream"] == "σ70 -10 site"

    legend = legend_entries_for_record(record)
    legend_labels = {label for _, label in legend}
    assert "σ70 -35 site" in legend_labels
    assert "σ70 -10 site" in legend_labels


def test_densegen_adapter_reads_fixed_elements_from_used_tfbs_detail() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAATTTGGGCCCCCCAAAATTTGGG",
        "densegen__used_tfbs_detail": [
            {"part_kind": "tfbs", "regulator": "lexA", "orientation": "fwd", "sequence": "AAA", "offset": 0},
            {
                "part_kind": "fixed_element",
                "role": "upstream",
                "constraint_name": "sigma70_core",
                "sequence": "TTTGGG",
                "offset": 5,
                "length": 6,
                "spacer_length": 6,
                "variant_id": "a",
                "placement_index": 0,
            },
            {
                "part_kind": "fixed_element",
                "role": "downstream",
                "constraint_name": "sigma70_core",
                "sequence": "AAAATT",
                "offset": 17,
                "length": 6,
                "spacer_length": 6,
                "variant_id": "H",
                "placement_index": 0,
            },
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
    assert len(record.features) == 3
    assert len(record.effects) == 1
    assert record.display.tag_labels["promoter:sigma70_core:upstream"] == "σ70 -35 site (a)"
    assert record.display.tag_labels["promoter:sigma70_core:downstream"] == "σ70 -10 site (H)"
    legend_labels = {label for _, label in legend_entries_for_record(record)}
    assert "σ70 -35 site (a)" in legend_labels
    assert "σ70 -10 site (H)" in legend_labels


def test_densegen_adapter_prefers_offset_raw_for_fixed_elements_from_used_tfbs_detail() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAATTTGGGCCCCCCAAAATTTGGG",
        "densegen__used_tfbs_detail": [
            {"part_kind": "tfbs", "regulator": "lexA", "orientation": "fwd", "sequence": "AAA", "offset": 0},
            {
                "part_kind": "fixed_element",
                "role": "upstream",
                "constraint_name": "sigma70_core",
                "sequence": "TTTGGG",
                "offset": 6,
                "offset_raw": 5,
                "pad_left": 1,
                "length": 6,
                "spacer_length": 6,
                "variant_id": "a",
                "placement_index": 0,
            },
            {
                "part_kind": "fixed_element",
                "role": "downstream",
                "constraint_name": "sigma70_core",
                "sequence": "AAAATT",
                "offset": 18,
                "offset_raw": 17,
                "pad_left": 1,
                "length": 6,
                "spacer_length": 6,
                "variant_id": "H",
                "placement_index": 0,
            },
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
    by_feature_id = {feature.id: feature for feature in record.features}
    assert by_feature_id["row1:promoter:sigma70_core:0:upstream"].span.start == 5
    assert by_feature_id["row1:promoter:sigma70_core:0:downstream"].span.start == 17


def test_densegen_adapter_appends_variant_ids_to_promoter_legend_labels() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAATTTGGGCCCCCCAAAATTTGGG",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "AAA", "offset": 0},
        ],
        "densegen__promoter_detail": {
            "placements": [
                {
                    "name": "sigma70_core",
                    "upstream_seq": "TTTGGG",
                    "downstream_seq": "AAAATT",
                    "upstream_start": 5,
                    "downstream_start": 17,
                    "spacer_length": 6,
                    "variant_ids": {"up_id": "a", "down_id": "H"},
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
    assert record.display.tag_labels["promoter:sigma70_core:upstream"] == "σ70 -35 site (a)"
    assert record.display.tag_labels["promoter:sigma70_core:downstream"] == "σ70 -10 site (H)"

    promoter_features = [feature for feature in record.features if feature.attrs.get("source") == "densegen_promoter"]
    assert len(promoter_features) == 2
    by_component = {str(feature.attrs.get("component")): feature for feature in promoter_features}
    assert by_component["upstream"].attrs.get("variant_id") == "a"
    assert by_component["downstream"].attrs.get("variant_id") == "H"

    legend = legend_entries_for_record(record)
    legend_labels = {label for _, label in legend}
    assert "σ70 -35 site (a)" in legend_labels
    assert "σ70 -10 site (H)" in legend_labels


def test_densegen_adapter_emits_promoter_spacer_effect_with_shared_track() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAATTTGGGCCCCCCAAAATTTGGG",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "AAA", "offset": 0},
        ],
        "densegen__promoter_detail": {
            "placements": [
                {
                    "name": "sigma70_core",
                    "upstream_seq": "TTTGGG",
                    "downstream_seq": "AAAATT",
                    "upstream_start": 5,
                    "downstream_start": 17,
                    "spacer_length": 6,
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
    assert len(record.effects) == 1
    effect = record.effects[0]
    assert effect.kind == "span_link"
    assert effect.params.get("label") == "6 bp"
    assert effect.params.get("lane") == "top"
    assert int(effect.render.get("track", -1)) == 0
    assert effect.target.get("from_feature_id") == "row1:promoter:sigma70_core:0:upstream"
    assert effect.target.get("to_feature_id") == "row1:promoter:sigma70_core:0:downstream"

    promoter_features = [feature for feature in record.features if feature.attrs.get("source") == "densegen_promoter"]
    assert len(promoter_features) == 2
    assert all(int(feature.render.get("track", -1)) == 0 for feature in promoter_features)

    style = resolve_style(preset=None, overrides={})
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    assert fig is not None
    plt.close(fig)


def test_densegen_adapter_promoter_spacer_length_mismatch_is_fatal() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAATTTGGGCCCCCCAAAATTTGGG",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "AAA", "offset": 0},
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

    with pytest.raises(SchemaError, match="spacer_length"):
        adapter.apply(row, row_index=0)


def test_densegen_palette_keeps_promoter_components_visually_related() -> None:
    palette = Palette({})
    upstream = palette.color_for("promoter:sigma70_core:upstream")
    downstream = palette.color_for("promoter:sigma70_core:downstream")
    tf = palette.color_for("tf:lexA")

    def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    assert _distance(upstream, downstream) < _distance(upstream, tf)


def test_densegen_spacer_annotation_is_centered_on_promoter_feature_track() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAATTTGGGCCCCCCAAAATTTGGG",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "AAA", "offset": 0},
        ],
        "densegen__promoter_detail": {
            "placements": [
                {
                    "name": "sigma70_core",
                    "upstream_seq": "TTTGGG",
                    "downstream_seq": "AAAATT",
                    "upstream_start": 5,
                    "downstream_start": 17,
                    "spacer_length": 6,
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
    style = resolve_style(
        preset="presentation_default",
        overrides={"show_reverse_complement": False, "connectors": False},
    )
    layout = compute_layout(record, style)
    expected_y = layout.y_forward + layout.feature_track_base_offset_up

    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        axis = fig.axes[0]
        span_labels = [text for text in axis.texts if text.get_text() == "6 bp"]
        assert len(span_labels) == 1
        assert float(span_labels[0].get_position()[1]) == pytest.approx(expected_y)

        rounded_boxes = [patch for patch in axis.patches if isinstance(patch, FancyBboxPatch)]
        assert rounded_boxes
        assert any(float(patch.get_linewidth()) > 0.0 for patch in rounded_boxes)
        assert any(
            tuple(float(channel) for channel in patch.get_edgecolor()[:3])
            != tuple(float(channel) for channel in patch.get_facecolor()[:3])
            for patch in rounded_boxes
        )
    finally:
        plt.close(fig)


def test_densegen_spacer_interval_reserves_promoter_track_for_span_annotation() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAATTTGGGAACCAAAAAATTGGG",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "CCAA", "offset": 13},
        ],
        "densegen__promoter_detail": {
            "placements": [
                {
                    "name": "sigma70_core",
                    "upstream_seq": "TTTGGG",
                    "downstream_seq": "AAAATT",
                    "upstream_start": 5,
                    "downstream_start": 17,
                    "spacer_length": 6,
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
    style = resolve_style(
        preset="presentation_default",
        overrides={"show_reverse_complement": False, "connectors": False},
    )
    layout = compute_layout(record, style)

    promoter_track = int(layout.feature_track_by_id["row1:promoter:sigma70_core:0:upstream"])
    tf_track = int(layout.feature_track_by_id["row1:tf:lexA:0"])
    assert promoter_track == 0
    assert tf_track != promoter_track


def test_densegen_fixed_element_annotations_prefer_top_lane_before_side_fallback() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAATTTGGGCCCCCCAAAATTTGGG",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "AAA", "offset": 0},
        ],
        "densegen__promoter_detail": {
            "placements": [
                {
                    "name": "sigma70_core",
                    "upstream_seq": "TTTGGG",
                    "downstream_seq": "AAAATT",
                    "upstream_start": 5,
                    "downstream_start": 17,
                    "spacer_length": 6,
                    "variant_ids": {"up_id": "a", "down_id": "H"},
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
    style = resolve_style(
        preset="presentation_default",
        overrides={"show_reverse_complement": False, "connectors": False},
    )
    layout = compute_layout(record, style)
    upstream_box = layout.feature_boxes["row1:promoter:sigma70_core:0:upstream"]
    upstream_y = float((upstream_box[1] + upstream_box[3]) / 2.0)

    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        axis = fig.axes[0]
        annotation_texts = [
            text
            for text in axis.texts
            if text.get_text() == "-35 site (a)" and float(text.get_position()[1]) > upstream_y
        ]
        assert annotation_texts
        assert str(annotation_texts[0].get_ha()).lower() == "center"
        span_labels = [text for text in axis.texts if text.get_text() == "6 bp"]
        assert span_labels
        downstream_labels = [text for text in axis.texts if text.get_text() == "-10 site (H)"]
        assert downstream_labels
        span_font_size = float(span_labels[0].get_fontsize())
        assert float(annotation_texts[0].get_fontsize()) == pytest.approx(span_font_size)
        assert float(downstream_labels[0].get_fontsize()) == pytest.approx(span_font_size)
    finally:
        plt.close(fig)


def test_densegen_layout_keeps_strand_centerline_balanced_with_asymmetric_feature_density() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAACCCCCGGGGGTTTTTACGTACGATCGA",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "AAAAAC", "offset": 0},
            {"regulator": "AraC", "orientation": "fwd", "sequence": "GGGGGT", "offset": 10},
            {"regulator": "LacI", "orientation": "fwd", "sequence": "ACGTAC", "offset": 20},
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
    style = resolve_style(
        preset="presentation_default",
        overrides={"show_reverse_complement": True, "connectors": True},
    )
    layout = compute_layout(record, style)
    centerline = (float(layout.y_forward) + float(layout.y_reverse)) / 2.0
    top_extent = float(layout.content_top) - centerline
    bottom_extent = centerline - float(layout.content_bottom)
    assert top_extent == pytest.approx(bottom_extent, abs=1e-6)


def test_densegen_layout_centers_strands_in_plot_when_overlay_is_absent() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAACCCCCGGGGGTTTTTACGTACGATCGA",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "AAAAAC", "offset": 0},
            {"regulator": "AraC", "orientation": "rev", "sequence": "CGATCG", "offset": 25},
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
    style = resolve_style(
        preset="presentation_default",
        overrides={"show_reverse_complement": True, "connectors": True, "legend": False},
    )
    layout = compute_layout(record, style)
    centerline = (float(layout.y_forward) + float(layout.y_reverse)) / 2.0
    plot_center = float(layout.height) / 2.0
    assert centerline == pytest.approx(plot_center, abs=1e-6)


def test_densegen_layout_centers_strands_in_plot_when_overlay_is_absent_with_bottom_legend() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAACCCCCGGGGGTTTTTACGTACGATCGA",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "AAAAAC", "offset": 0},
            {"regulator": "AraC", "orientation": "rev", "sequence": "CGATCG", "offset": 25},
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
    style = resolve_style(
        preset="presentation_default",
        overrides={"show_reverse_complement": True, "connectors": True, "legend": True, "legend_mode": "bottom"},
    )
    layout = compute_layout(record, style)
    centerline = (float(layout.y_forward) + float(layout.y_reverse)) / 2.0
    plot_center = float(layout.height) / 2.0
    assert centerline == pytest.approx(plot_center, abs=1e-6)


def test_densegen_bottom_legend_stays_within_plot_width_with_large_requested_gap() -> None:
    sequence = ("AAAAAA" + ("T" * 14) + "CCCCCC" + ("T" * 14) + "GGGGGG" + ("T" * 34)).upper()
    row = {
        "id": "row1",
        "sequence": sequence,
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "AAAAAA", "offset": 0},
            {"regulator": "AraC", "orientation": "fwd", "sequence": "CCCCCC", "offset": 20},
            {"regulator": "LacI", "orientation": "fwd", "sequence": "GGGGGG", "offset": 40},
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
    style = resolve_style(
        preset="presentation_default",
        overrides={
            "show_reverse_complement": True,
            "legend": True,
            "legend_mode": "bottom",
            "legend_gap_x": 420.0,
        },
    )
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        axis = fig.axes[0]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_patches = [
            patch for patch in axis.patches if isinstance(patch, FancyBboxPatch) and float(patch.get_zorder()) == 10.0
        ]
        legend_texts = [text for text in axis.texts if float(text.get_zorder()) == 10.0]
        assert legend_patches, "Expected bottom-legend patches."
        assert legend_texts, "Expected bottom-legend labels."
        max_patch_x = max(float(patch.get_x() + patch.get_width()) for patch in legend_patches)
        max_text_x = max(
            float(axis.transData.inverted().transform((text.get_window_extent(renderer=renderer).x1, 0.0))[0])
            for text in legend_texts
        )
        x_max = float(axis.get_xlim()[1])
        assert max(max_patch_x, max_text_x) <= (x_max + 1e-6)
    finally:
        plt.close(fig)


def test_presentation_default_renders_solid_strand_connectors() -> None:
    row = {
        "id": "row1",
        "sequence": "ACGTACGT",
        "densegen__used_tfbs_detail": [],
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
    style = resolve_style(
        preset="presentation_default",
        overrides={"show_reverse_complement": True, "connectors": True},
    )
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        axis = fig.axes[0]
        connector_lines = []
        for line in axis.lines:
            xdata = line.get_xdata(orig=False)
            ydata = line.get_ydata(orig=False)
            if len(xdata) != 2 or len(ydata) != 2:
                continue
            if abs(float(xdata[0]) - float(xdata[1])) >= 1e-9:
                continue
            connector_lines.append(line)

        assert len(connector_lines) == len(record.sequence)
        assert all(not line.is_dashed() for line in connector_lines)
    finally:
        plt.close(fig)


def test_presentation_default_centers_connector_span_between_strands() -> None:
    row = {
        "id": "row1",
        "sequence": "ACGTACGT",
        "densegen__used_tfbs_detail": [],
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
    style = resolve_style(
        preset="presentation_default",
        overrides={"show_reverse_complement": True, "connectors": True},
    )
    layout = compute_layout(record, style)
    top_row_boundary = float(layout.y_forward) - float(layout.sequence_extent_down)
    bottom_row_boundary = float(layout.y_reverse) + float(layout.sequence_extent_up)
    available_gap = top_row_boundary - bottom_row_boundary
    expected_mid = (top_row_boundary + bottom_row_boundary) / 2.0
    expected_span = available_gap * 0.5

    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        axis = fig.axes[0]
        connector_lines = []
        for line in axis.lines:
            xdata = line.get_xdata(orig=False)
            ydata = line.get_ydata(orig=False)
            if len(xdata) != 2 or len(ydata) != 2:
                continue
            if abs(float(xdata[0]) - float(xdata[1])) >= 1e-9:
                continue
            connector_lines.append(line)

        assert connector_lines
        first = connector_lines[0]
        y_start, y_end = [float(value) for value in first.get_ydata(orig=False)]
        observed_mid = (y_start + y_end) / 2.0
        observed_span = abs(y_end - y_start)
        assert observed_mid == pytest.approx(expected_mid, abs=1e-6)
        assert observed_span == pytest.approx(expected_span, abs=1e-6)
    finally:
        plt.close(fig)


def test_densegen_layout_avoids_same_lane_for_adjacent_boxes_when_padding_would_overlap() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAAACCCCCC",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "AAAAAA", "offset": 0},
            {"regulator": "araC", "orientation": "fwd", "sequence": "CCCCCC", "offset": 6},
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
    style = resolve_style(
        preset="presentation_default",
        overrides={"show_reverse_complement": False, "connectors": False},
    )
    layout = compute_layout(record, style)
    tracks = {placement.feature_id: int(placement.track) for placement in layout.placements}
    assert tracks["row1:tf:lexA:0"] != tracks["row1:tf:araC:1"]


def test_densegen_fixed_element_annotations_do_not_overlap_promoter_span_line() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAATTTGGGCAAAAATTGGGG",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "AAAAA", "offset": 0},
        ],
        "densegen__promoter_detail": {
            "placements": [
                {
                    "name": "sigma70_core",
                    "upstream_seq": "TTTGGG",
                    "downstream_seq": "AAAATT",
                    "upstream_start": 5,
                    "downstream_start": 13,
                    "spacer_length": 2,
                    "variant_ids": {
                        "up_id": "a",
                        "down_id": "H",
                    },
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
    style = resolve_style(
        preset="presentation_default",
        overrides={"show_reverse_complement": False, "connectors": False},
    )
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        axis = fig.axes[0]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        fixed_texts = [
            text
            for text in axis.texts
            if text.get_text().startswith("-35 site") or text.get_text().startswith("-10 site")
        ]
        assert fixed_texts

        span_segments: list[tuple[float, float, float]] = []
        for line in axis.lines:
            x_data = line.get_xdata(orig=False)
            y_data = line.get_ydata(orig=False)
            if len(x_data) != 2 or len(y_data) != 2:
                continue
            x0 = float(x_data[0])
            x1 = float(x_data[1])
            y0 = float(y_data[0])
            y1 = float(y_data[1])
            if abs(y0 - y1) > 1e-6:
                continue
            if abs(x1 - x0) <= 1e-6:
                continue
            span_segments.append((min(x0, x1), max(x0, x1), y0))

        assert span_segments
        for text in fixed_texts:
            text_bbox = text.get_window_extent(renderer=renderer).transformed(axis.transData.inverted())
            tx0 = float(text_bbox.x0)
            tx1 = float(text_bbox.x1)
            ty0 = float(text_bbox.y0)
            ty1 = float(text_bbox.y1)
            for sx0, sx1, sy in span_segments:
                overlaps_x = max(tx0, sx0) < min(tx1, sx1)
                overlaps_y = ty0 < (sy + 2.0) and ty1 > (sy - 2.0)
                assert not (overlaps_x and overlaps_y)
    finally:
        plt.close(fig)


def test_densegen_span_link_occupied_boxes_include_line_endpoints() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAATTTGGGCCCAAAATTGGGG",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "AAAAA", "offset": 0},
        ],
        "densegen__promoter_detail": {
            "placements": [
                {
                    "name": "sigma70_core",
                    "upstream_seq": "TTTGGG",
                    "downstream_seq": "AAAATT",
                    "upstream_start": 5,
                    "downstream_start": 14,
                    "spacer_length": 3,
                    "variant_ids": {"up_id": "a", "down_id": "H"},
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
    style = resolve_style(
        preset="presentation_default",
        overrides={"show_reverse_complement": False, "connectors": False},
    )
    layout = compute_layout(record, style)
    boxes = _span_link_label_boxes(record, layout, style)
    assert boxes

    up_box = layout.feature_boxes["row1:promoter:sigma70_core:0:upstream"]
    down_box = layout.feature_boxes["row1:promoter:sigma70_core:0:downstream"]
    inner_margin_px = float(style.span_link_inner_margin_bp) * float(layout.cw)
    span_x1 = float(up_box[2]) + inner_margin_px
    span_x2 = float(down_box[0]) - inner_margin_px
    span_y = float(layout.y_forward) + float(layout.feature_track_base_offset_up)

    assert any(
        float(bx0) <= span_x1 <= float(bx1) and float(by0) <= span_y <= float(by1) for bx0, by0, bx1, by1 in boxes
    )
    assert any(
        float(bx0) <= span_x2 <= float(bx1) and float(by0) <= span_y <= float(by1) for bx0, by0, bx1, by1 in boxes
    )


def test_densegen_overlapping_promoter_region_preserves_internal_kmer_text_labels() -> None:
    row = {
        "id": "row1",
        "sequence": "AAAAATTTGGGCCCCCCAAAATTTGGG",
        "densegen__used_tfbs_detail": [
            {"regulator": "background", "orientation": "fwd", "sequence": "GGGCCC", "offset": 8},
        ],
        "densegen__promoter_detail": {
            "placements": [
                {
                    "name": "sigma70_core",
                    "upstream_seq": "TTTGGG",
                    "downstream_seq": "AAAATT",
                    "upstream_start": 5,
                    "downstream_start": 17,
                    "spacer_length": 6,
                    "variant_ids": {"up_id": "d", "down_id": "B"},
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
    style = resolve_style(
        preset="presentation_default",
        overrides={"show_reverse_complement": False, "connectors": False},
    )
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        axis = fig.axes[0]
        white_feature_text = [
            patch
            for patch in axis.patches
            if isinstance(patch, PathPatch)
            and int(round(float(patch.get_zorder()))) == 4
            and all(abs(float(channel) - 1.0) < 1e-6 for channel in patch.get_facecolor()[:3])
        ]
        assert white_feature_text
    finally:
        plt.close(fig)
