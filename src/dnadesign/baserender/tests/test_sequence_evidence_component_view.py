"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_sequence_evidence_component_view.py

Component-unit sequence evidence rendering tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import FancyBboxPatch

from dnadesign.baserender.src.adapters import build_adapter
from dnadesign.baserender.src.config import AdapterCfg, resolve_style
from dnadesign.baserender.src.core import SchemaError
from dnadesign.baserender.src.render import Palette, render_record
from dnadesign.baserender.src.render.layout import compute_layout
from dnadesign.baserender.src.runtime import initialize_runtime


def _component_span_contract() -> dict[str, object]:
    return {
        "contract_kind": "sequence_evidence_map_v1",
        "state_id": "component_unit.component_span_qa",
        "topology_kind": "linear_ssdna",
        "alphabet": "dna",
        "primary_sequence": "ACGT",
        "complement_sequence": "TGCA",
        "owners": [
            {
                "owner_id": "unit.left_segment",
                "row_id": "primary",
                "start": 0,
                "end": 2,
                "display_label": "left_segment",
                "short_label": "",
            },
            {
                "owner_id": "unit.right_segment",
                "row_id": "primary",
                "start": 2,
                "end": 4,
                "display_label": "right_segment",
                "short_label": "",
            },
        ],
        "effect_tags": [
            {
                "tag_id": "unit.stem_base",
                "tag_kind": "stem_base",
                "row_id": "primary",
                "start": 1,
                "end": 3,
                "display_label": "stem_base",
                "short_label": "",
            }
        ],
        "boundaries": [],
        "pairings": [
            {
                "pairing_id": "unit.payload_rc",
                "primary_start": 0,
                "primary_end": 2,
                "complement_start": 2,
                "complement_end": 4,
                "display_label": "payload_rc",
                "short_label": "intended RC",
            }
        ],
        "display": {"title": "component span QA"},
        "meta": {
            "interval_annotation_policy": "span_backdrops_only",
            "render_pairing_links": False,
            "row_labels": {"primary": "Top", "complement": "Bottom"},
            "span_backdrops": [
                {
                    "start": 0,
                    "end": 2,
                    "cover_rows": "both",
                    "fill": "#A7F3D0",
                    "alpha": 0.32,
                    "corner_radius": 2.0,
                },
                {
                    "start": 2,
                    "end": 4,
                    "cover_rows": "both",
                    "fill": "#BFDBFE",
                    "alpha": 0.32,
                    "corner_radius": 2.0,
                },
            ],
            "segment_labels": [
                {"text": "Left segment", "start": 0, "end": 2, "label_side": "above"},
                {"text": "Right segment", "start": 2, "end": 4, "label_side": "above"},
                {"text": "Stem base", "start": 1, "end": 3, "label_side": "below", "color": "#475569"},
            ],
            "segment_label_gap_px": 6.0,
            "segment_label_tier_gap_px": 10.0,
        },
    }


def test_sequence_evidence_component_view_uses_backdrops_not_annotation_boxes() -> None:
    adapter = build_adapter(AdapterCfg(kind="sequence_evidence_map_v1", columns={}, policies={}), alphabet="DNA")

    record = adapter.apply(_component_span_contract(), row_index=0)

    assert record.features == ()
    assert record.effects == ()
    assert record.meta["row_labels"] == {"primary": "Top", "complement": "Bottom"}
    assert record.meta["segment_labels"][-1]["label_side"] == "below"
    assert all(backdrop["cover_rows"] == "both" for backdrop in record.meta["span_backdrops"])


def test_sequence_evidence_component_view_fails_fast_on_invalid_render_policy() -> None:
    adapter = build_adapter(AdapterCfg(kind="sequence_evidence_map_v1", columns={}, policies={}), alphabet="DNA")
    payload = _component_span_contract()
    meta = dict(payload["meta"])
    meta["interval_annotation_policy"] = "annotation_boxes_and_backdrops"
    payload["meta"] = meta

    with pytest.raises(SchemaError, match="interval_annotation_policy"):
        adapter.apply(payload, row_index=0)


def test_sequence_evidence_component_view_renders_pairing_dashes_and_duplex_backdrops() -> None:
    adapter = build_adapter(AdapterCfg(kind="sequence_evidence_map_v1", columns={}, policies={}), alphabet="DNA")
    record = adapter.apply(_component_span_contract(), row_index=0)
    style = resolve_style(
        preset="presentation_default",
        overrides={
            "show_reverse_complement": True,
            "connectors": True,
            "baseline_spacing": 34.0,
            "color_ticks": "#CBD5E1",
            "connector_alpha": 0.42,
            "connector_width": 0.5,
            "connector_dash": [1.0, 2.2],
            "legend": False,
            "legend_mode": "none",
            "font_size_seq": 10,
            "font_size_label": 9,
            "sequence": {"strand_gap_cells": 0.16},
        },
    )
    layout = compute_layout(record, style)

    assert layout.y_forward - layout.y_reverse < 40.0

    initialize_runtime()
    fig = render_record(record, renderer_name="nucleotide_evidence_map", style=style, palette=Palette(style.palette))
    try:
        axis = fig.axes[0]
        connector_lines = [
            line for line in axis.lines if str(line.get_gid() or "").startswith("sequence_pair_connector:")
        ]
        assert len(connector_lines) == len(record.sequence)
        assert {line.get_color() for line in connector_lines} == {"#CBD5E1"}
        backdrops = [
            patch
            for patch in axis.patches
            if isinstance(patch, FancyBboxPatch) and str(patch.get_gid() or "").startswith("sequence_backdrop:")
        ]
        assert len(backdrops) == 2
        for patch in backdrops:
            bbox = patch.get_bbox()
            assert bbox.y0 <= layout.y_reverse - layout.sequence_extent_down
            assert bbox.y1 >= layout.y_forward + layout.sequence_extent_up
        text_by_label = {text.get_text(): text for text in axis.texts}
        assert {"Left segment", "Right segment", "Stem base"}.issubset(text_by_label)
        assert text_by_label["Left segment"].get_position()[1] - layout.y_forward < 20.0
        assert layout.y_reverse - text_by_label["Stem base"].get_position()[1] < 18.0
        label_boxes = []
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for label in ("Left segment", "Right segment", "Stem base"):
            label_boxes.append(text_by_label[label].get_window_extent(renderer=renderer))
        for index, box in enumerate(label_boxes):
            for other in label_boxes[index + 1 :]:
                assert not box.overlaps(other)
    finally:
        plt.close(fig)
