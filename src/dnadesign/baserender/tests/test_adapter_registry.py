"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_adapter_registry.py

Adapter registry tests for centralized factory and required-source-column contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import FancyBboxPatch

from dnadesign.baserender.src.adapters import build_adapter, required_source_columns
from dnadesign.baserender.src.adapters.cruncher_best_window import CruncherBestWindowAdapter
from dnadesign.baserender.src.adapters.duplex_sequence_v1 import DuplexSequenceV1Adapter
from dnadesign.baserender.src.adapters.hairpin_topology_v1 import HairpinTopologyV1Adapter
from dnadesign.baserender.src.adapters.scar_nick_visual_v1 import ScarNickVisualV1Adapter
from dnadesign.baserender.src.adapters.sequence_evidence_map_v1 import (
    SequenceEvidenceMapV1Adapter,
    _style_token_for_owner,
    _style_token_for_tag,
)
from dnadesign.baserender.src.adapters.sequence_windows_v1 import SequenceWindowsV1Adapter
from dnadesign.baserender.src.adapters.snapback_visual_v1 import SnapbackVisualV1Adapter
from dnadesign.baserender.src.adapters.yiu_hairpin_topology_v1 import (
    YiuHairpinTopologyV1Adapter,
)
from dnadesign.baserender.src.adapters.yiu_hairpin_topology_v1 import (
    _span as _yiu_hairpin_span,
)
from dnadesign.baserender.src.adapters.yiu_linear_state_v1 import YiuLinearStateV1Adapter
from dnadesign.baserender.src.adapters.yiu_topology_cartoon_v1 import YiuTopologyCartoonV1Adapter
from dnadesign.baserender.src.config import AdapterCfg, resolve_style
from dnadesign.baserender.src.config.adapter_contracts import adapter_descriptor
from dnadesign.baserender.src.core import SchemaError
from dnadesign.baserender.src.render import Palette, legend_entries_for_record, render_record
from dnadesign.baserender.src.runtime import initialize_runtime

from .conftest import write_parquet


def test_required_source_columns_densegen_includes_optional_present_columns() -> None:
    cfg = AdapterCfg(
        kind="densegen_tfbs",
        columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
            "overlay_text": "details",
            "video_subtitle": "subtitle",
        },
        policies={},
    )
    assert required_source_columns(cfg) == ["sequence", "densegen__used_tfbs_detail", "id", "details", "subtitle"]


def test_required_source_columns_generic_features_omits_missing_optional_columns() -> None:
    cfg = AdapterCfg(
        kind="generic_features",
        columns={
            "sequence": "sequence",
            "features": "features",
        },
        policies={},
    )
    assert required_source_columns(cfg) == ["sequence", "features"]


def test_required_source_columns_unknown_kind_is_schema_error() -> None:
    cfg = AdapterCfg(kind="unknown_kind", columns={}, policies={})
    with pytest.raises(SchemaError, match="Unsupported adapter kind"):
        required_source_columns(cfg)


def test_required_source_columns_missing_required_key_is_schema_error() -> None:
    cfg = AdapterCfg(
        kind="densegen_tfbs",
        columns={"annotations": "densegen__used_tfbs_detail"},
        policies={},
    )
    with pytest.raises(SchemaError, match="missing required adapter column key"):
        required_source_columns(cfg)


def test_required_source_columns_densegen_accepts_overlay_text_optional_key() -> None:
    cfg = AdapterCfg(
        kind="densegen_tfbs",
        columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
            "overlay_text": "details",
        },
        policies={},
    )
    assert required_source_columns(cfg) == ["sequence", "densegen__used_tfbs_detail", "id", "details"]


def test_required_source_columns_densegen_accepts_video_subtitle_optional_key() -> None:
    cfg = AdapterCfg(
        kind="densegen_tfbs",
        columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
            "video_subtitle": "subtitle",
        },
        policies={},
    )
    assert required_source_columns(cfg) == ["sequence", "densegen__used_tfbs_detail", "id", "subtitle"]


def test_generic_features_adapter_accepts_display_video_subtitle() -> None:
    cfg = AdapterCfg(
        kind="generic_features",
        columns={
            "id": "id",
            "sequence": "sequence",
            "features": "features",
            "display": "display",
        },
        policies={},
    )
    adapter = build_adapter(cfg, alphabet="IUPAC_DNA")
    record = adapter.apply(
        {
            "id": "row-1",
            "sequence": "ACGT",
            "features": [],
            "display": {
                "overlay_text": None,
                "tag_labels": {"tf:lexA": "lexA"},
                "video_subtitle": "lexA=0.80 cpxR=0.71",
            },
        },
        row_index=0,
    )
    assert record.display.video_subtitle == "lexA=0.80 cpxR=0.71"


def test_sequence_evidence_map_adapter_requires_no_source_columns() -> None:
    cfg = AdapterCfg(kind="sequence_evidence_map_v1", columns={}, policies={})

    assert required_source_columns(cfg) == []

    adapter = build_adapter(cfg, alphabet="DNA")
    assert isinstance(adapter, SequenceEvidenceMapV1Adapter)


def test_snapback_visual_adapter_requires_no_source_columns() -> None:
    cfg = AdapterCfg(kind="snapback_visual_v1", columns={}, policies={})

    assert required_source_columns(cfg) == []

    adapter = build_adapter(cfg, alphabet="DNA")
    assert isinstance(adapter, SnapbackVisualV1Adapter)


def _scar_nick_adapter_payload() -> dict[str, object]:
    pre_sequence = "GGTCTCGGCCC"
    pre_complement = "CCAGAGCCGGG"
    post_complement = "CCAGAGCCTGT"
    spacer = "NNNN"
    post_offset = len(pre_sequence) + len(spacer)
    pre_panel = {
        "panel_id": "pre_release",
        "title": "before terminal nick",
        "state_kind": "pre_terminal_nick",
        "nick_state": "intact",
        "start": 0,
        "end": len(pre_sequence),
        "terminal_boundary": 11,
        "nick_boundary": 11,
        "retained_product_span": {"start": 7, "end": 11},
        "release_site_span": {"start": 0, "end": 6},
        "type_iis_offset_span": {"start": 6, "end": 7},
        "retained_scar_span": {"start": 7, "end": 11},
        "nickase_site_span": {"start": 0, "end": 11},
        "fragment_spans": [],
    }
    post_panel = {
        "panel_id": "post_release",
        "title": "after terminal nick",
        "state_kind": "post_terminal_nick",
        "nick_state": "nicked",
        "start": post_offset,
        "end": post_offset + len(pre_sequence),
        "terminal_boundary": post_offset + 11,
        "nick_boundary": post_offset + 11,
        "retained_product_span": {"start": post_offset + 7, "end": post_offset + 11},
        "release_site_span": {"start": post_offset, "end": post_offset + 6},
        "type_iis_offset_span": {"start": post_offset + 6, "end": post_offset + 7},
        "retained_scar_span": {"start": post_offset + 7, "end": post_offset + 11},
        "nickase_site_span": {"start": post_offset, "end": post_offset + 11},
        "fragment_spans": [{"row": "complement", "start": post_offset, "end": post_offset + 11}],
    }
    fills = []
    for panel in (pre_panel, post_panel):
        prefix = panel["panel_id"]
        fill_specs = [
            ("type_iis_release_site", "type_iis_release_site", "release_site_span", "#F0E442", 0.34),
            ("retained_type_iis_scar", "retained_type_iis_scar", "retained_scar_span", "#009E73", 0.36),
        ]
        if panel["panel_id"] == "pre_release":
            fill_specs.append(("nickase_footprint", "nickase_footprint", "nickase_site_span", "#56B4E9", 0.24))
        for fill_id, semantic, span_name, color, alpha in fill_specs:
            span = panel[span_name]
            fills.append(
                {
                    "fill_id": f"{prefix}_{fill_id}",
                    "semantic": semantic,
                    "start": span["start"],
                    "end": span["end"],
                    "cover_rows": "primary"
                    if panel["panel_id"] == "post_release" and semantic == "retained_type_iis_scar"
                    else "both",
                    "fill": color,
                    "alpha": alpha,
                    "corner_radius": 0.0,
                }
            )
        if panel["panel_id"] == "post_release":
            fragment_span = panel["fragment_spans"][0]
            fills.append(
                {
                    "fill_id": f"{prefix}_annealed_adapter_fragment_0",
                    "semantic": "annealed_adapter_fragment",
                    "start": fragment_span["start"],
                    "end": fragment_span["end"],
                    "cover_rows": fragment_span["row"],
                    "fill": "#CBD5E1",
                    "alpha": 0.48,
                    "corner_radius": 4.0,
                    "edge_color": "#94A3B8",
                    "edge_alpha": 0.64,
                    "edge_linewidth": 0.45,
                }
            )
        for position in range(panel["retained_scar_span"]["start"], panel["retained_scar_span"]["end"]):
            for row_id in ("primary", "complement"):
                fills.append(
                    {
                        "fill_id": f"{prefix}_degenerate_nucleotide_{row_id}_{position}",
                        "semantic": "degenerate_nucleotide",
                        "start": position,
                        "end": position + 1,
                        "cover_rows": row_id,
                        "fill": "#E0F2FE",
                        "alpha": 0.84,
                        "corner_radius": 3.0,
                        "edge_color": "#93C5FD",
                        "edge_alpha": 0.80,
                        "edge_linewidth": 0.36,
                    }
                )
    return {
        "contract_kind": "scar_nick_visual_v1",
        "state_id": "candidate_01.pre_post_terminal_nick",
        "state_kind": "pre_post_terminal_nick",
        "event_scope": "terminal_nick",
        "alphabet": "iupac_dna",
        "primary_sequence": pre_sequence + spacer + pre_sequence,
        "complement_sequence": pre_complement + spacer + post_complement,
        "primary_row_label": "Top",
        "complement_row_label": "Bottom",
        "terminal_boundary": post_offset + 11,
        "nick_boundary": post_offset + 11,
        "retained_product_span": {"start": post_offset + 7, "end": post_offset + 11},
        "release_site_span": {"start": post_offset, "end": post_offset + 6},
        "type_iis_offset_span": {"start": post_offset + 6, "end": post_offset + 7},
        "retained_scar_span": {"start": post_offset + 7, "end": post_offset + 11},
        "junction_partner_span": None,
        "nickase_site_span": {"start": post_offset, "end": post_offset + 11},
        "nickase_site_source_span": {"start": -7, "end": 4},
        "nick_state": "pre_post",
        "retained_scar": "GCCC",
        "left_base": "GCCC",
        "right_base": "TGTC",
        "nicked_strand": "bottom",
        "surviving_strand": "top",
        "profile_s3s2s1s0": "MXMX",
        "profile_payload_outward": "XMXM",
        "pair_classes": [
            {
                "position": 0,
                "site": "S3",
                "source_offset": 0,
                "left_base": "G",
                "right_base": "C",
                "aligned_right_base": "G",
                "class_label": "M",
            },
            {
                "position": 1,
                "site": "S2",
                "source_offset": 1,
                "left_base": "C",
                "right_base": "T",
                "aligned_right_base": "A",
                "class_label": "X",
            },
            {
                "position": 2,
                "site": "S1",
                "source_offset": 2,
                "left_base": "C",
                "right_base": "G",
                "aligned_right_base": "C",
                "class_label": "M",
            },
            {
                "position": 3,
                "site": "S0",
                "source_offset": 3,
                "left_base": "C",
                "right_base": "T",
                "aligned_right_base": "A",
                "class_label": "X",
            },
        ],
        "panels": [pre_panel, post_panel],
        "rectangular_fills": fills,
        "release_placement": {
            "variant_id": "BsaI-HFv2",
            "orientation": "forward",
            "recognition_sequence": "GGTCTC",
            "recognition_site_excised": True,
            "source_catalog_id": "type_iis_release_v1",
            "source_url": "https://www.neb.com/en-us/products/r3733-bsai-hf-v2",
            "commercial_confidence": "primary_vendor_current",
            "warning_codes": [],
            "recognition_site_start": -7,
            "recognition_site_end": -1,
            "top_cut_boundary": 0,
            "bottom_cut_boundary": 4,
            "retained_scar_start": 0,
            "retained_scar_end": 4,
            "retained_scar_nt": 4,
        },
        "nickase": {
            "variant_id": "Test.TerminalBottomNickase",
            "specificity_id": "TerminalBottomNickase",
            "orientation": "forward",
            "canonical_read_row": "primary",
            "motif_top_5to3": "GGTCTCGNNNN",
            "canonical_motif_top_5to3": "GGTCTCGNNNN",
            "recognition_nt": 7,
            "vendor": "dnadesign test fixture",
            "source_url": "https://example.invalid/dnadesign/scar-nick-terminal-fixture",
            "source_family": "nicking_endonuclease",
            "commercial_confidence": "primary_vendor_current",
            "warning_codes": [],
            "site": "Test.TerminalBottomNickase:forward[-7,4)",
            "source_site_start": -7,
            "source_site_end": 4,
            "strand": "bottom",
            "boundary": 4,
            "terminal_boundary": 4,
            "display_boundary": post_offset + 11,
            "display_site_span": {"start": post_offset, "end": post_offset + 11},
            "exact_terminal": True,
        },
        "meta": {
            "panel_spacer_indices": list(range(len(pre_sequence), post_offset)),
            "mismatch_indices": [post_offset + 8, post_offset + 10],
        },
    }


def test_scar_nick_visual_adapter_maps_rectangular_scar_fill_to_evidence_backdrop() -> None:
    cfg = AdapterCfg(kind="scar_nick_visual_v1", columns={}, policies={})

    assert required_source_columns(cfg) == []

    adapter = build_adapter(cfg, alphabet="DNA")
    assert isinstance(adapter, ScarNickVisualV1Adapter)

    record = adapter.apply(_scar_nick_adapter_payload(), row_index=0)

    assert record.id == "candidate_01.pre_post_terminal_nick"
    assert record.meta["adapter"] == "scar_nick_visual_v1"
    assert record.meta["span_backdrops"][0]["start"] == 0
    assert record.meta["span_backdrops"][0]["end"] == 6
    assert record.meta["span_backdrops"][0]["corner_radius"] == 0.0
    assert record.features == ()
    assert "before terminal nick" not in record.display.overlay_text
    assert "after terminal nick" not in record.display.overlay_text
    assert record.meta["segment_labels"][0]["text"] == "BsaI-HFv2 GGTCTC"
    assert record.meta["segment_labels"][1]["text"] == "Test.TerminalBottomNickase GGTCTCGNNNN"
    assert record.meta["segment_labels"][1]["row_id"] == "primary"
    assert record.meta["segment_labels"][1]["label_side"] == "below"
    assert record.meta["segment_labels"][2]["text"] == "BsaI-HFv2 GGTCTC"
    assert record.meta["segment_labels"][3]["text"] == "Y adaptor"
    assert record.meta["segment_labels"][3]["row_id"] == "complement"
    assert record.meta["segment_labels"][3]["label_side"] == "below"
    assert {label["text"] for label in record.meta["segment_labels"]} == {
        "BsaI-HFv2 GGTCTC",
        "Test.TerminalBottomNickase GGTCTCGNNNN",
        "Y adaptor",
    }
    assert len(record.meta["segment_labels"]) == 4
    assert record.meta["panel_transition_arrows"] == [{"start": 11, "end": 15}]
    assert not any(
        backdrop["semantic"] == "nickase_footprint" and backdrop["start"] >= 15
        for backdrop in record.meta["span_backdrops"]
    )
    assert record.meta["base_highlights"]["primary"] == list(range(0, 11))
    assert record.meta["base_highlights"]["complement"] == []
    assert record.meta["base_highlight_colors"]["primary"][0] == "#7A6500"
    assert record.meta["base_highlight_colors"]["primary"][6] == "#005A8D"
    assert record.meta["dim_base_indices"]["complement"] == list(range(15, 26))
    degenerate_fills = [
        backdrop for backdrop in record.meta["span_backdrops"] if backdrop["semantic"] == "degenerate_nucleotide"
    ]
    assert len(degenerate_fills) == 16
    assert {
        (backdrop["start"], backdrop["end"], backdrop["cover_rows"], backdrop["corner_radius"])
        for backdrop in degenerate_fills
    } >= {
        (7, 8, "primary", 3.0),
        (7, 8, "complement", 3.0),
        (22, 23, "primary", 3.0),
        (22, 23, "complement", 3.0),
    }
    fragment_fills = [
        backdrop for backdrop in record.meta["span_backdrops"] if backdrop["semantic"] == "annealed_adapter_fragment"
    ]
    assert fragment_fills == [
        {
            "semantic": "annealed_adapter_fragment",
            "start": 15,
            "end": 26,
            "fill": "#CBD5E1",
            "alpha": 0.48,
            "corner_radius": 4.0,
            "cover_rows": "complement",
            "edge_color": "#94A3B8",
            "edge_alpha": 0.64,
            "edge_linewidth": 0.45,
        }
    ]
    assert record.meta["base_dim_color"] == "#94A3B8"
    assert record.meta["connector_hidden_indices"] == [11, 12, 13, 14]
    assert record.meta["connector_cross_indices"] == [23, 25]
    assert record.meta["connector_cross_color"] == "#111827"
    assert record.meta["cell_width_scale"] == 1.12
    assert record.meta["span_edge_markers"][0]["start"] == 0
    assert record.meta["span_edge_markers"][0]["end"] == 6
    assert record.meta["span_edge_markers"][-1]["start"] == 15
    assert record.meta["panel_spans"][0]["panel_id"] == "pre_release"
    assert record.meta["grid_max_rows"] == 5
    assert len(record.effects) == 2
    assert all(effect.kind == "boundary_marker" for effect in record.effects)
    assert [effect.target for effect in record.effects] == [
        {"boundary": 11, "lane": "complement"},
        {"boundary": 26, "lane": "complement"},
    ]
    assert all(effect.params["label"] == "" for effect in record.effects)
    assert record.meta["scar_nick"]["profile_order"] == "S3_S2_S1_S0"
    assert record.meta["scar_nick"]["type_iis_recognition_sequence"] == "GGTCTC"
    assert record.meta["scar_nick"]["nickase_motif_top_5to3"] == "GGTCTCGNNNN"
    assert record.meta["scar_nick"]["nickase_canonical_motif_top_5to3"] == "GGTCTCGNNNN"

    second_row = adapter.apply(_scar_nick_adapter_payload(), row_index=1)
    assert "before terminal nick" not in second_row.display.overlay_text
    assert second_row.meta["segment_labels"][0]["text"] == "BsaI-HFv2 GGTCTC"


def test_scar_nick_annealed_adapter_backdrop_renders_thin_edge() -> None:
    cfg = AdapterCfg(kind="scar_nick_visual_v1", columns={}, policies={})
    adapter = build_adapter(cfg, alphabet="DNA")
    record = adapter.apply(_scar_nick_adapter_payload(), row_index=0)
    annealed_index = next(
        index
        for index, backdrop in enumerate(record.meta["span_backdrops"])
        if backdrop["semantic"] == "annealed_adapter_fragment"
    )

    style = resolve_style(preset=None, overrides={"connectors": True})
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        patch_by_gid = {patch.get_gid(): patch for patch in fig.axes[0].patches if patch.get_gid()}
        backdrop = patch_by_gid[f"sequence_backdrop:{annealed_index}"]
    finally:
        plt.close(fig)

    assert isinstance(backdrop, FancyBboxPatch)
    assert mcolors.to_hex(backdrop.get_edgecolor(), keep_alpha=False) == "#94a3b8"
    assert float(backdrop.get_edgecolor()[3]) == pytest.approx(0.64)
    assert float(backdrop.get_linewidth()) == pytest.approx(0.45)


def test_scar_nick_top_strand_y_adaptor_label_does_not_collide_with_title() -> None:
    payload = _scar_nick_adapter_payload()
    payload["title"] = "01 | L=AGTG/R=TCTA | WXMM"
    payload["nicked_strand"] = "top"
    payload["surviving_strand"] = "bottom"
    payload["nickase"]["strand"] = "top"
    payload["panels"][1]["fragment_spans"][0]["row"] = "primary"
    for fill in payload["rectangular_fills"]:
        if fill["semantic"] == "annealed_adapter_fragment":
            fill["cover_rows"] = "primary"
        if fill["fill_id"] == "post_release_retained_type_iis_scar":
            fill["cover_rows"] = "complement"

    cfg = AdapterCfg(kind="scar_nick_visual_v1", columns={}, policies={})
    adapter = build_adapter(cfg, alphabet="DNA")
    record = adapter.apply(payload, row_index=0)

    y_adaptor_label = next(label for label in record.meta["segment_labels"] if label["text"] == "Y adaptor")
    assert y_adaptor_label["row_id"] == "primary"

    style = resolve_style(
        preset=None,
        overrides={
            "legend": False,
            "figure_scale": 1.0,
            "font_mono": "DejaVu Sans Mono",
            "font_label": "DejaVu Sans Mono",
            "font_size_seq": 12,
            "font_size_label": 8,
            "font_size_span_link_label": 8,
            "padding_x": 34.0,
            "padding_y": 48.0,
            "baseline_spacing": 48.0,
            "overlay_align": "center",
            "overlay_title_color": "#4B5563",
        },
    )
    palette = Palette(style.palette)
    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=palette)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        y_adaptor_bbox = next(
            text.get_window_extent(renderer=renderer) for text in fig.axes[0].texts if text.get_text() == "Y adaptor"
        )
        title_bbox = next(
            text.get_window_extent(renderer=renderer)
            for text in fig.axes[0].texts
            if text.get_text() == payload["title"]
        )
    finally:
        plt.close(fig)

    assert not (
        max(y_adaptor_bbox.x0, title_bbox.x0) < min(y_adaptor_bbox.x1, title_bbox.x1)
        and max(y_adaptor_bbox.y0, title_bbox.y0) < min(y_adaptor_bbox.y1, title_bbox.y1)
    )


def test_scar_nick_visual_adapter_places_nickase_label_on_nicked_top_strand() -> None:
    payload = _scar_nick_adapter_payload()
    payload["nicked_strand"] = "top"
    payload["surviving_strand"] = "bottom"
    payload["nickase"]["strand"] = "top"
    payload["panels"][1]["fragment_spans"][0]["row"] = "primary"
    for fill in payload["rectangular_fills"]:
        if fill["semantic"] == "annealed_adapter_fragment":
            fill["cover_rows"] = "primary"
        if fill["fill_id"] == "post_release_retained_type_iis_scar":
            fill["cover_rows"] = "complement"
    cfg = AdapterCfg(kind="scar_nick_visual_v1", columns={}, policies={})
    adapter = build_adapter(cfg, alphabet="DNA")

    record = adapter.apply(payload, row_index=0)

    assert record.meta["segment_labels"][1]["row_id"] == "primary"
    assert record.meta["segment_labels"][1]["label_side"] == "below"
    assert [effect.target for effect in record.effects] == [
        {"boundary": 11, "lane": "primary"},
        {"boundary": 26, "lane": "primary"},
    ]


def test_scar_nick_visual_adapter_bolds_reverse_nickase_on_canonical_complement_row() -> None:
    payload = _scar_nick_adapter_payload()
    payload["nicked_strand"] = "top"
    payload["surviving_strand"] = "bottom"
    payload["nickase"]["strand"] = "top"
    payload["nickase"]["orientation"] = "reverse"
    payload["nickase"]["canonical_read_row"] = "complement"
    payload["nickase"]["canonical_motif_top_5to3"] = "NNNNCGAGACC"
    payload["panels"][1]["fragment_spans"][0]["row"] = "primary"
    for fill in payload["rectangular_fills"]:
        if fill["semantic"] == "annealed_adapter_fragment":
            fill["cover_rows"] = "primary"
        if fill["fill_id"] == "post_release_retained_type_iis_scar":
            fill["cover_rows"] = "complement"
    cfg = AdapterCfg(kind="scar_nick_visual_v1", columns={}, policies={})
    adapter = build_adapter(cfg, alphabet="DNA")

    record = adapter.apply(payload, row_index=0)

    assert record.meta["segment_labels"][1]["row_id"] == "complement"
    assert record.meta["segment_labels"][1]["text"] == "Test.TerminalBottomNickase NNNNCGAGACC"
    assert record.meta["base_highlights"]["primary"] == list(range(0, 6))
    assert record.meta["base_highlights"]["complement"] == list(range(0, 11))
    assert record.meta["base_highlight_colors"]["complement"][0] == "#005A8D"
    assert [effect.target for effect in record.effects] == [
        {"boundary": 11, "lane": "primary"},
        {"boundary": 26, "lane": "primary"},
    ]


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    [
        ("duplex_sequence_v1", DuplexSequenceV1Adapter),
        ("hairpin_topology_v1", HairpinTopologyV1Adapter),
        ("sequence_windows_v1", SequenceWindowsV1Adapter),
        ("yiu_linear_state_v1", YiuLinearStateV1Adapter),
        ("yiu_hairpin_topology_v1", YiuHairpinTopologyV1Adapter),
        ("yiu_topology_cartoon_v1", YiuTopologyCartoonV1Adapter),
    ],
)
def test_build_adapter_constructs_contract_driven_adapter_types(kind: str, expected_type: type[object]) -> None:
    cfg = AdapterCfg(kind=kind, columns={}, policies={})

    adapter = build_adapter(cfg, alphabet="DNA")

    assert isinstance(adapter, expected_type)


def test_adapter_descriptor_policy_normalizers_accept_supported_values() -> None:
    densegen_policies = adapter_descriptor("densegen_tfbs").normalize_policies(
        {
            "ambiguous": "first",
            "offset_mode": "one_based",
            "on_missing_kmer": "skip_entry",
            "on_invalid_row": "skip",
            "min_per_record": "2",
            "require_non_null_cols": ["sequence", 7],
            "zero_as_unspecified": False,
            "require_non_empty": True,
            "overlay_text_template": "{overlay_text}\n{id}",
        },
        "input.adapter.policies",
    )
    cruncher_policies = adapter_descriptor("cruncher_best_window").normalize_policies(
        {"on_missing_hit": "skip", "on_missing_pwm": "skip_effect"},
        "input.adapter.policies",
    )

    assert densegen_policies == {
        "ambiguous": "first",
        "offset_mode": "one_based",
        "on_missing_kmer": "skip_entry",
        "on_invalid_row": "skip",
        "min_per_record": 2,
        "require_non_null_cols": ["sequence", "7"],
        "zero_as_unspecified": False,
        "require_non_empty": True,
        "overlay_text_template": "{overlay_text}\n{id}",
    }
    assert cruncher_policies == {"on_missing_hit": "skip", "on_missing_pwm": "skip_effect"}


def test_densegen_policy_normalizer_rejects_invalid_non_null_cols_type() -> None:
    with pytest.raises(SchemaError, match="require_non_null_cols must be a list"):
        adapter_descriptor("densegen_tfbs").normalize_policies(
            {"require_non_null_cols": "sequence"},
            "input.adapter.policies",
        )


def test_densegen_policy_normalizer_rejects_non_bool_flags() -> None:
    with pytest.raises(SchemaError, match="zero_as_unspecified must be bool"):
        adapter_descriptor("densegen_tfbs").normalize_policies(
            {"zero_as_unspecified": "yes"},
            "input.adapter.policies",
        )


def test_densegen_policy_normalizer_rejects_blank_overlay_template() -> None:
    with pytest.raises(SchemaError, match="overlay_text_template must be a non-empty string"):
        adapter_descriptor("densegen_tfbs").normalize_policies(
            {"overlay_text_template": " "},
            "input.adapter.policies",
        )


def test_build_adapter_constructs_cruncher_adapter_from_existing_inputs(tmp_path) -> None:
    hits_path = write_parquet(
        tmp_path / "hits.parquet",
        [
            {
                "elite_id": "elite-1",
                "tf": "lexA",
                "best_start": 1,
                "best_strand": "+",
                "best_window_seq": "TTGACA",
                "best_core_seq": "TTGACA",
            }
        ],
    )
    config_path = tmp_path / "cruncher.yaml"
    config_path.write_text(
        "cruncher:\n  pwms_info:\n    lexA:\n      pwm_matrix:\n        - [0.25, 0.25, 0.25, 0.25]\n"
    )
    cfg = AdapterCfg(
        kind="cruncher_best_window",
        columns={
            "sequence": "sequence",
            "id": "id",
            "hits_path": str(hits_path),
            "config_path": str(config_path),
        },
        policies={},
    )

    adapter = build_adapter(cfg, alphabet="DNA")

    assert isinstance(adapter, CruncherBestWindowAdapter)


@pytest.mark.parametrize(
    ("owner_id", "expected"),
    [
        ("payload_left_half", "segment_payload"),
        ("snapback_adapter", "segment_adapter"),
        ("source_forward_primer_binding_region", "segment_primer"),
        ("retained_region", "segment_retained"),
        ("sacrificial_region_long", "segment_sacrificial"),
        ("misc_region", "segment"),
    ],
)
def test_sequence_evidence_map_owner_style_tokens(owner_id: str, expected: str) -> None:
    assert _style_token_for_owner(owner_id) == expected


@pytest.mark.parametrize(
    ("tag_kind", "expected"),
    [
        ("payload_overhang_left", "site_overhang"),
        ("type_iis_recognition_left", "site_recognition"),
        ("primer_bindable_by_source_forward", "site_primer"),
        ("adapter_binding", "site_adapter"),
        ("ligation_junction_member", "site_boundary"),
        ("custom_effect", "site_effect"),
    ],
)
def test_sequence_evidence_map_tag_style_tokens(tag_kind: str, expected: str) -> None:
    assert _style_token_for_tag(tag_kind) == expected


def test_sequence_evidence_map_adapter_applies_contract_without_complement_sequence() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(
        {
            "contract_kind": "sequence_evidence_map_v1",
            "state_id": "state-1",
            "topology_kind": "linear_dsdna",
            "alphabet": "dna",
            "primary_sequence": "ACGTACGTACGT",
            "owners": [
                {
                    "owner_id": "payload_left_half",
                    "row_id": "primary",
                    "start": 0,
                    "end": 4,
                    "display_label": "Payload",
                    "short_label": "PAY",
                },
                {
                    "owner_id": "retained_region",
                    "row_id": "primary",
                    "start": 4,
                    "end": 6,
                    "display_label": "Retained",
                    "short_label": "RET",
                },
                {
                    "owner_id": "source_forward_primer_binding_region",
                    "row_id": "primary",
                    "start": 6,
                    "end": 8,
                    "display_label": "Primer",
                    "short_label": "PRI",
                },
                {
                    "owner_id": "snapback_adapter",
                    "row_id": "primary",
                    "start": 8,
                    "end": 10,
                    "display_label": "Adapter",
                    "short_label": "ADP",
                },
                {
                    "owner_id": "sacrificial_region_long",
                    "row_id": "primary",
                    "start": 10,
                    "end": 12,
                    "display_label": "Sacrificial",
                    "short_label": "SAC",
                },
            ],
            "effect_tags": [
                {
                    "tag_id": "recognition",
                    "tag_kind": "type_iis_recognition_left",
                    "row_id": "primary",
                    "start": 0,
                    "end": 2,
                    "display_label": "Recognition",
                    "short_label": "REC",
                },
                {
                    "tag_id": "primer",
                    "tag_kind": "primer_bindable_by_source_forward",
                    "row_id": "primary",
                    "start": 2,
                    "end": 4,
                    "display_label": "Primer",
                    "short_label": "PRI",
                },
                {
                    "tag_id": "adapter",
                    "tag_kind": "adapter_binding",
                    "row_id": "primary",
                    "start": 4,
                    "end": 6,
                    "display_label": "Adapter",
                    "short_label": "ADP",
                },
                {
                    "tag_id": "junction",
                    "tag_kind": "ligation_junction_member",
                    "row_id": "primary",
                    "start": 6,
                    "end": 8,
                    "display_label": "Junction",
                    "short_label": "JCT",
                },
                {
                    "tag_id": "custom",
                    "tag_kind": "custom_effect",
                    "row_id": "primary",
                    "start": 8,
                    "end": 10,
                    "display_label": "Custom",
                    "short_label": "CUS",
                },
            ],
            "boundaries": [
                {
                    "boundary_id": "cut-1",
                    "row_id": "primary",
                    "boundary": 6,
                    "boundary_kind": "cut",
                    "display_label": "Cut",
                    "short_label": "CUT",
                }
            ],
            "pairings": [
                {
                    "pairing_id": "pair-1",
                    "primary_start": 0,
                    "primary_end": 2,
                    "complement_start": 10,
                    "complement_end": 12,
                    "display_label": "Pair",
                    "short_label": "PAIR",
                }
            ],
            "display": {"title": "Example"},
            "meta": {"mode": "test"},
        },
        row_index=0,
    )

    assert record.meta["show_reverse_complement"] is False
    assert record.features[0].attrs["style_token"] == "segment_payload"
    assert record.features[1].attrs["style_token"] == "segment_retained"
    assert record.features[2].attrs["style_token"] == "segment_primer"
    assert record.features[3].attrs["style_token"] == "segment_adapter"
    assert record.features[4].attrs["style_token"] == "segment_sacrificial"
    assert record.features[5].attrs["style_token"] == "site_recognition"
    assert record.features[6].attrs["style_token"] == "site_primer"
    assert record.features[7].attrs["style_token"] == "site_adapter"
    assert record.features[8].attrs["style_token"] == "site_boundary"
    assert record.features[9].attrs["style_token"] == "site_effect"
    assert [effect.kind for effect in record.effects] == ["boundary_marker", "span_link"]
    assert record.effects[0].params["semantic"] == "cut"


def test_sequence_evidence_map_adapter_can_exclude_tags_from_legend() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(
        {
            "contract_kind": "sequence_evidence_map_v1",
            "state_id": "assembled_payload",
            "topology_kind": "linear_ssdna",
            "alphabet": "iupac_dna",
            "primary_sequence": "CTCTATATCTGATATAGAG",
            "owners": [
                {
                    "owner_id": "payload_left_half",
                    "row_id": "primary",
                    "start": 0,
                    "end": 9,
                    "display_label": "Left payload half",
                    "short_label": "L",
                },
                {
                    "owner_id": "payload_right_half",
                    "row_id": "primary",
                    "start": 9,
                    "end": 19,
                    "display_label": "Right payload half",
                    "short_label": "R",
                },
            ],
            "effect_tags": [
                {
                    "tag_id": "bulge_2",
                    "tag_kind": "payload_bulge_position",
                    "row_id": "primary",
                    "start": 10,
                    "end": 11,
                    "display_label": "Bulge 2",
                    "short_label": "B2",
                },
            ],
            "boundaries": [],
            "pairings": [],
            "display": {"title": "Assembled payload"},
            "meta": {
                "legend_exclude_tags": [
                    "owner:payload_left_half",
                    "owner:payload_right_half",
                ]
            },
        },
        row_index=0,
    )

    assert record.meta["legend_exclude_tags"] == (
        "owner:payload_left_half",
        "owner:payload_right_half",
    )
    assert legend_entries_for_record(record) == [
        ("effect:payload_bulge_position", "Bulge 2"),
    ]


def test_sequence_evidence_map_adapter_places_hairpin_pairings_on_bottom_lane() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(
        {
            "contract_kind": "sequence_evidence_map_v1",
            "state_id": "snapback_foldback",
            "topology_kind": "hairpin_folded",
            "alphabet": "dna",
            "primary_sequence": "TCAGCAGTCTTGACT",
            "complement_sequence": "AGTCGTCAGAACTGA",
            "owners": [],
            "effect_tags": [],
            "boundaries": [],
            "pairings": [
                {
                    "pairing_id": "pair-1",
                    "primary_start": 5,
                    "primary_end": 6,
                    "complement_start": 11,
                    "complement_end": 12,
                    "display_label": "WC pair",
                    "short_label": "",
                }
            ],
            "display": {"title": "Foldback"},
            "meta": {},
        },
        row_index=0,
    )

    assert [effect.kind for effect in record.effects] == ["span_link"]
    assert record.effects[0].params["lane"] == "bottom"
    assert record.effects[0].params["label"] == ""
    assert record.effects[0].render["track"] == 0


def test_sequence_evidence_map_adapter_preserves_explicit_complement_and_base_highlights() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(
        {
            "contract_kind": "sequence_evidence_map_v1",
            "state_id": "assembled_payload",
            "topology_kind": "linear_dsdna",
            "alphabet": "iupac_dna",
            "primary_sequence": "CTCTATATCTGATATAGAG",
            "complement_sequence": "GAGATATAGAATATATCTC",
            "owners": [],
            "effect_tags": [],
            "boundaries": [
                {
                    "boundary_id": "left-overhang-boundary",
                    "row_id": "primary",
                    "boundary": 9,
                    "boundary_kind": "ligation_junction",
                    "display_label": "",
                    "short_label": "",
                },
            ],
            "pairings": [],
            "display": {"title": "Assembled payload"},
            "meta": {
                "base_highlight_color": "#B91C1C",
                "base_highlights": {
                    "primary": [10],
                    "complement": [10],
                },
                "dim_base_indices": {
                    "primary": [0, 1, 2, 3, 4, 5, 6],
                    "complement": [0, 1, 2, 3, 4, 5, 6],
                },
                "connector_hidden_indices": [9, 11, 12],
                "connector_cross_indices": [10],
                "connector_overhang_spans": [{"start": 9, "end": 13}],
                "segment_labels": [
                    {"text": "Left", "start": 0, "end": 9},
                    {"text": "Right", "start": 9, "end": 19},
                ],
            },
        },
        row_index=0,
    )

    assert record.meta["show_reverse_complement"] is True
    assert record.meta["complement_sequence"] == "GAGATATAGAATATATCTC"
    assert record.meta["base_highlight_color"] == {"primary": "#B91C1C", "complement": "#B91C1C"}
    assert record.meta["base_highlights"] == {"primary": (10,), "complement": (10,)}
    assert record.meta["dim_base_indices"] == {
        "primary": (0, 1, 2, 3, 4, 5, 6),
        "complement": (0, 1, 2, 3, 4, 5, 6),
    }
    assert record.meta["connector_hidden_indices"] == (9, 11, 12)
    assert record.meta["connector_cross_indices"] == (10,)
    assert record.meta["connector_overhang_spans"] == ({"start": 9, "end": 13},)
    assert record.meta["segment_labels"] == (
        {"text": "Left", "start": 0, "end": 9, "row_id": "primary"},
        {"text": "Right", "start": 9, "end": 19, "row_id": "primary"},
    )
    boundary_effects = [effect for effect in record.effects if effect.kind == "boundary_marker"]
    assert len(boundary_effects) == 1
    assert boundary_effects[0].target == {"boundary": 9, "lane": "primary"}


def test_snapback_visual_adapter_embeds_contract_for_snapback_renderer() -> None:
    adapter = SnapbackVisualV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(
        {
            "contract_kind": "snapback_visual_v1",
            "state_id": "demo.post_nick_foldback",
            "state_kind": "post_nick_foldback",
            "alphabet": "dna",
            "title": "Foldback",
            "primary_sequence": "TCAGCAGTCTTGACTA",
            "complement_sequence": "AGTCGTCAGAACTGAT",
            "primary_row_label": "Foldback",
            "complement_row_label": "Partner",
            "ligation_junction_boundary": 5,
            "released_prefix_span": {"start": 0, "end": 5},
            "retained_stem_span": {"start": 5, "end": 9},
            "cap_span": {"start": 9, "end": 12},
            "foldback_revcomp_span": {"start": 12, "end": 16},
            "loop_geometry": {
                "kind": "hairpin_corner_triloop_v1",
                "source_cap_span": {"start": 9, "end": 11},
                "cap_extension_span": {"start": 11, "end": 12},
                "display_primary_span": {"start": 5, "end": 9},
                "display_complement_span": {"start": 12, "end": 16},
            },
            "pairings": [
                {"left_index": 5, "right_index": 15},
                {"left_index": 6, "right_index": 14},
            ],
            "primary_mismatch_positions": [6],
            "complement_mismatch_positions": [14],
            "meta": {"source_view_kind": "snapback_post_nick_foldback_v1"},
        },
        row_index=0,
    )

    assert record.display.overlay_text is None
    assert record.effects == ()
    assert record.features == ()
    assert record.meta["contract"]["state_kind"] == "post_nick_foldback"
    assert record.meta["contract"]["pairings"] == [
        {"left_index": 5, "right_index": 15},
        {"left_index": 6, "right_index": 14},
    ]
    assert record.meta["contract"]["loop_geometry"]["kind"] == "hairpin_corner_triloop_v1"


def test_sequence_evidence_map_adapter_normalizes_span_backdrops() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(
        {
            "contract_kind": "sequence_evidence_map_v1",
            "state_id": "assembled_payload",
            "topology_kind": "linear_dsdna",
            "alphabet": "iupac_dna",
            "primary_sequence": "CTCTATATCTGATATAGAG",
            "complement_sequence": "GAGATATAGAATATATCTC",
            "owners": [],
            "effect_tags": [],
            "boundaries": [],
            "pairings": [],
            "display": {"title": "Assembled payload"},
            "meta": {
                "span_backdrops": [
                    {
                        "semantic": "stem_base_left",
                        "start": 9,
                        "end": 13,
                        "coordinate_space": "payload_forward",
                        "fill": "#BFDBFE",
                        "alpha": 0.3,
                        "corner_radius": 8.0,
                        "cover_rows": "both",
                        "edge_color": "#2563EB",
                        "edge_alpha": 0.72,
                        "edge_linewidth": 0.5,
                    }
                ]
            },
        },
        row_index=0,
    )

    assert record.meta["span_backdrops"] == (
        {
            "semantic": "stem_base_left",
            "start": 9,
            "end": 13,
            "coordinate_space": "payload_forward",
            "fill": "#BFDBFE",
            "alpha": 0.3,
            "corner_radius": 8.0,
            "cover_rows": "both",
            "edge_color": "#2563EB",
            "edge_alpha": 0.72,
            "edge_linewidth": 0.5,
        },
    )


def test_sequence_evidence_map_adapter_rejects_legacy_boundary_marker_style_meta() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    with pytest.raises(SchemaError, match="meta.boundary_marker_style is no longer supported"):
        adapter.apply(
            {
                "contract_kind": "sequence_evidence_map_v1",
                "state_id": "legacy-boundary-style",
                "topology_kind": "linear_dsdna",
                "alphabet": "iupac_dna",
                "primary_sequence": "AACCGGTT",
                "owners": [],
                "effect_tags": [],
                "boundaries": [],
                "pairings": [],
                "display": {"title": "Legacy"},
                "meta": {"boundary_marker_style": "dashed_uncapped"},
            },
            row_index=0,
        )


def test_sequence_evidence_map_adapter_rejects_connector_indices_outside_overhang_spans() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    with pytest.raises(SchemaError, match="connector_cross_indices must lie within connector_overhang_spans"):
        adapter.apply(
            {
                "contract_kind": "sequence_evidence_map_v1",
                "state_id": "bad-overhang-cross",
                "topology_kind": "linear_dsdna",
                "alphabet": "iupac_dna",
                "primary_sequence": "AACCGGTT",
                "owners": [],
                "effect_tags": [],
                "boundaries": [
                    {
                        "boundary_id": "left",
                        "row_id": "primary",
                        "boundary": 2,
                        "boundary_kind": "cut",
                        "display_label": "",
                        "short_label": "",
                    },
                    {
                        "boundary_id": "right",
                        "row_id": "complement",
                        "boundary": 6,
                        "boundary_kind": "cut",
                        "display_label": "",
                        "short_label": "",
                    },
                ],
                "pairings": [],
                "display": {"title": "Bad"},
                "meta": {
                    "connector_overhang_spans": [{"start": 2, "end": 6}],
                    "connector_cross_indices": [1],
                },
            },
            row_index=0,
        )


def test_sequence_evidence_map_adapter_allows_connector_spans_without_matching_boundary_positions() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(
        {
            "contract_kind": "sequence_evidence_map_v1",
            "state_id": "assembled-payload-single-seam",
            "topology_kind": "linear_dsdna",
            "alphabet": "iupac_dna",
            "primary_sequence": "AACCGGTT",
            "owners": [],
            "effect_tags": [],
            "boundaries": [
                {
                    "boundary_id": "join",
                    "row_id": "primary",
                    "boundary": 4,
                    "boundary_kind": "ligation_junction",
                    "display_label": "",
                    "short_label": "",
                },
            ],
            "pairings": [],
            "display": {"title": "Assembled payload"},
            "meta": {
                "connector_overhang_spans": [{"start": 2, "end": 6}],
                "connector_hidden_indices": [2, 3, 5],
                "connector_cross_indices": [4],
            },
        },
        row_index=0,
    )

    assert record.meta["connector_overhang_spans"] == ({"start": 2, "end": 6},)
    boundary_effects = [effect for effect in record.effects if effect.kind == "boundary_marker"]
    assert len(boundary_effects) == 1
    assert boundary_effects[0].target == {"boundary": 4, "lane": "primary"}


def test_sequence_evidence_map_adapter_rejects_invalid_contract_payload() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    with pytest.raises(SchemaError, match="Invalid sequence_evidence_map_v1 contract at row 7"):
        adapter.apply(
            {
                "contract_kind": "sequence_evidence_map_v1",
                "state_id": "bad-state",
                "topology_kind": "linear_dsdna",
                "alphabet": "dna",
                "primary_sequence": "",
            },
            row_index=7,
        )


def test_sequence_evidence_map_adapter_rejects_invalid_record_after_contract_validation() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="DNA")

    with pytest.raises(SchemaError, match="Sequence contains invalid characters for DNA"):
        adapter.apply(
            {
                "contract_kind": "sequence_evidence_map_v1",
                "state_id": "bad-sequence",
                "topology_kind": "linear_dsdna",
                "alphabet": "dna",
                "primary_sequence": "ACGU",
            },
            row_index=3,
        )


def _linear_duplex_payload(*, sequence: str = "TTTACCTCAGCAAAGCTGAGGTAAA") -> dict:
    return {
        "version": 1,
        "kind": "linear_duplex_v1",
        "view_id": "hit_001.linear_duplex",
        "solution_id": "abc123def456",
        "title": "Hit 1 - Linear duplex",
        "coordinate_semantics": "boundary_inclusive_v2",
        "primary_sequence_5to3": sequence,
        "sequence_span": {"start": 0, "end": len(sequence)},
        "cassette_span": {"start": 0, "end": len(sequence)},
        "row_labels": {
            "primary": "5' -> 3' primary",
            "complement": "3' -> 5' complement",
        },
        "target_strand": "complement",
        "segments": [
            {"id": "stem5p_arm", "start": 0, "end": 10, "semantic": "stem5p_arm", "label": "Stem 5' arm"},
            {"id": "loop", "start": 10, "end": 15, "semantic": "loop", "label": "Loop"},
            {"id": "stem3p_arm", "start": 15, "end": 25, "semantic": "stem3p_arm", "label": "Stem 3' arm"},
        ],
        "site_instances": [
            {
                "id": "left_site",
                "variant_id": "Nb.BbvCI",
                "specificity_id": "BbvCI",
                "start": 2,
                "end": 9,
                "orientation": "forward",
                "intent": "intended_left",
                "label": "Nb.BbvCI",
                "site_target_strand": "complement",
            }
        ],
        "nick_events": [
            {
                "id": "left_nick",
                "boundary": 7,
                "target_strand": "complement",
                "source_site_id": "left_site",
                "intent": "intended_left",
                "label": "Nick",
            }
        ],
        "bounded_segment": {
            "start_boundary": 7,
            "end_boundary": 20,
            "target_strand": "complement",
            "label": "Bounded nicked segment",
        },
        "labels": [{"text": "Target strand: complement", "placement": "header"}],
        "meta": {"rank": 1},
    }


def _hairpin_topology_payload(*, sequence: str = "ACCTCAGCAAAGCTGAGGT") -> dict:
    return {
        "version": 1,
        "kind": "ssdna_hairpin_v1",
        "view_id": "hit_001.ssdna_hairpin",
        "solution_id": "abc123def456",
        "title": "Hit 1 - ssDNA hairpin",
        "primary_sequence_5to3": sequence,
        "topology": {
            "stem5p_span": {"start": 0, "end": 7},
            "loop_span": {"start": 7, "end": 12},
            "stem3p_span": {"start": 12, "end": 19},
        },
        "pair_map": [
            {"left_index": 0, "right_index": 18},
            {"left_index": 1, "right_index": 17},
        ],
        "feature_spans": [
            {
                "id": "left_site_projection",
                "start": 1,
                "end": 7,
                "semantic": "motif_projection",
                "label": "Nb.BbvCI motif",
            }
        ],
        "duplex_derived_annotations": [
            {
                "kind": "informational_note",
                "text": "Nicking is defined in the linear duplex interpretation.",
            }
        ],
        "meta": {"rank": 1},
    }


def _yiu_linear_state_payload(*, sequence: str = "CCTCAGCCCGCTGATCCCTATCAGTGATAGA") -> dict:
    return {
        "contract_kind": "yiu_linear_state_v1",
        "state_id": "hairpin_pcr_linear_insert",
        "topology_kind": "linear_dsdna",
        "alphabet": "iupac_dna",
        "primary_sequence": sequence,
        "complement_sequence": "TCTATCACTGATAGGGATCAGCGGGCTGAGG",
        "segments": [
            {"segment_id": "left_arm", "state_start": 0, "state_end": 5},
            {"segment_id": "skip", "state_start": 5, "state_end": 5},
        ],
        "annotations": [],
        "cuts": [{"site_id": "cut-1", "top_boundary": 4, "bottom_boundary": 8}],
        "junctions": [{"id": "junction-1", "join_index": 12}],
        "fragments": [],
        "display": {"title": "Split-payload insert"},
        "meta": {"evidence_mode": "pattern_compatibility"},
    }


def _yiu_hairpin_topology_payload(*, sequence: str = "CCTCAGCCCGCTGATCAGCGGGCTGAGG") -> dict:
    return {
        "contract_kind": "yiu_hairpin_topology_v1",
        "state_id": "ligated_ssdna_hairpin",
        "topology_kind": "ssdna_hairpin",
        "sequence": sequence,
        "stem_left_span": {"start": 0, "end": 8},
        "stem_right_span": {"start": 20, "end": 28},
        "loop_span": {"start": 8, "end": 20},
        "pair_map": [{"left_index": 0, "right_index": 27}],
        "adapter_branches": [],
        "annotations": [{"note": "structured"}],
        "display": {"title": "Ligation hairpin"},
        "meta": {"evidence_mode": "concrete_realization"},
    }


def _yiu_topology_cartoon_payload(
    *,
    sequence: str = "CCGATGTCCCTATCAGTGATAGAGAGGGGGGGGGGGGCCTCAGCCCGCTGA",
) -> dict:
    return {
        "contract_kind": "yiu_topology_cartoon_v1",
        "state_id": "circularized_payload_candidate",
        "topology_kind": "circular_duplex",
        "sequence": sequence,
        "segments": [
            {"segment_id": "payload", "state_start": 0, "state_end": 10},
            {"segment_id": "skip", "state_start": 10, "state_end": 10},
        ],
        "annotations": [],
        "cuts": [],
        "junctions": [{"id": "junction", "join_index": 15}],
        "fragments": [],
        "display": {"title": "Circularized payload"},
        "meta": {"evidence_mode": "concrete_realization"},
    }


def test_duplex_sequence_adapter_applies_contract_payload() -> None:
    adapter = DuplexSequenceV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(_linear_duplex_payload(), row_index=0)

    assert record.id == "hit_001.linear_duplex"
    assert record.meta["adapter"] == "duplex_sequence_v1"
    assert record.meta["target_strand"] == "complement"
    assert record.display.tag_labels["bounded_segment"] == "Bounded nicked segment"


def test_hairpin_topology_adapter_applies_contract_payload() -> None:
    adapter = HairpinTopologyV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(_hairpin_topology_payload(), row_index=1)

    assert record.id == "hit_001.ssdna_hairpin"
    assert record.meta["adapter"] == "hairpin_topology_v1"
    assert record.meta["solution_id"] == "abc123def456"
    assert record.display.tag_labels["feature_projection"] == "Motif projection"


def test_hairpin_topology_adapter_wraps_invalid_contract_payload() -> None:
    adapter = HairpinTopologyV1Adapter(columns={}, policies={}, alphabet="DNA")

    with pytest.raises(SchemaError, match="Invalid ssdna_hairpin_v1 contract at row 2"):
        adapter.apply({"kind": "ssdna_hairpin_v1"}, row_index=2)


def test_yiu_linear_state_adapter_applies_contract_payload() -> None:
    adapter = YiuLinearStateV1Adapter(columns={}, policies={}, alphabet="IUPAC_DNA")

    record = adapter.apply(_yiu_linear_state_payload(), row_index=2)

    assert record.id == "hairpin_pcr_linear_insert"
    assert [feature.id for feature in record.features] == ["left_arm"]
    assert [effect.kind for effect in record.effects] == [
        "boundary_marker",
        "boundary_marker",
        "boundary_marker",
    ]
    assert record.meta["adapter"] == "yiu_linear_state_v1"


def test_yiu_hairpin_topology_adapter_applies_contract_payload() -> None:
    adapter = YiuHairpinTopologyV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(_yiu_hairpin_topology_payload(), row_index=3)

    assert record.id == "ligated_ssdna_hairpin"
    assert [feature.id for feature in record.features] == ["stem5p_span", "loop_span", "stem3p_span"]
    assert record.meta["adapter"] == "yiu_hairpin_topology_v1"
    assert record.meta["hairpin_notes"] == [{"note": "structured"}]


def test_yiu_topology_cartoon_adapter_applies_contract_payload() -> None:
    adapter = YiuTopologyCartoonV1Adapter(columns={}, policies={}, alphabet="DNA")

    record = adapter.apply(_yiu_topology_cartoon_payload(), row_index=4)

    assert record.id == "circularized_payload_candidate"
    assert [feature.id for feature in record.features] == ["payload"]
    assert record.meta["adapter"] == "yiu_topology_cartoon_v1"


@pytest.mark.parametrize(
    ("adapter", "payload"),
    [
        (
            DuplexSequenceV1Adapter(columns={}, policies={}, alphabet="DNA"),
            _linear_duplex_payload(sequence="TTRACCTCAGCAAAGCTGAGGTAAA"),
        ),
        (
            HairpinTopologyV1Adapter(columns={}, policies={}, alphabet="DNA"),
            _hairpin_topology_payload(sequence="ACRTCAGCAAAGCTGAGGT"),
        ),
        (
            YiuLinearStateV1Adapter(columns={}, policies={}, alphabet="DNA"),
            _yiu_linear_state_payload(sequence="CCTRAGCCCGCTGATCCCTATCAGTGATAGA"),
        ),
        (
            YiuHairpinTopologyV1Adapter(columns={}, policies={}, alphabet="DNA"),
            _yiu_hairpin_topology_payload(sequence="CCTRAGCCCGCTGATCAGCGGGCTGAGG"),
        ),
        (
            YiuTopologyCartoonV1Adapter(columns={}, policies={}, alphabet="DNA"),
            _yiu_topology_cartoon_payload(sequence="CCRATGTCCCTATCAGTGATAGAGAGGGGGGGGGGGGCCTCAGCCCGCTGA"),
        ),
    ],
)
def test_contract_driven_adapters_wrap_record_validation_errors(adapter: object, payload: dict) -> None:
    with pytest.raises(SchemaError, match="Sequence contains invalid characters for DNA"):
        adapter.apply(payload, row_index=5)


@pytest.mark.parametrize(
    "adapter",
    [
        YiuLinearStateV1Adapter(columns={}, policies={}, alphabet="DNA"),
        YiuHairpinTopologyV1Adapter(columns={}, policies={}, alphabet="DNA"),
        YiuTopologyCartoonV1Adapter(columns={}, policies={}, alphabet="DNA"),
    ],
)
def test_yiu_adapters_require_mapping_rows(adapter: object) -> None:
    with pytest.raises(SchemaError, match="row 6 must be a mapping"):
        adapter.apply("bad-row", row_index=6)


def test_yiu_topology_cartoon_adapter_wraps_invalid_contract_payload() -> None:
    adapter = YiuTopologyCartoonV1Adapter(columns={}, policies={}, alphabet="DNA")

    with pytest.raises(SchemaError, match="Invalid yiu_topology_cartoon_v1 contract at row 7"):
        adapter.apply({"contract_kind": "yiu_topology_cartoon_v1"}, row_index=7)


def test_yiu_hairpin_span_helper_rejects_invalid_bounds() -> None:
    assert _yiu_hairpin_span([2, 5], ctx="loop_span") == (2, 5)

    with pytest.raises(SchemaError, match="loop_span must be a 2-item list"):
        _yiu_hairpin_span([2], ctx="loop_span")

    with pytest.raises(SchemaError, match="loop_span end must be > start"):
        _yiu_hairpin_span([5, 5], ctx="loop_span")
