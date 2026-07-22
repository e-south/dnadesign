"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_usr_genbank_adapter.py

USR GenBank annotation adapter and render-contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from dnadesign.baserender.src.config import load_cruncher_showcase_job, resolve_style
from dnadesign.baserender.src.config.adapter_contracts import adapter_descriptor
from dnadesign.baserender.src.core import SchemaError
from dnadesign.baserender.src.public import adapt_record, get_render_contract_descriptor
from dnadesign.baserender.src.render import Palette, render_record
from dnadesign.baserender.src.render.layout import compute_layout
from dnadesign.baserender.src.runtime import initialize_runtime

from .conftest import write_job, write_parquet


def _genbank_row() -> dict[str, object]:
    return {
        "id": "seq1",
        "sequence": "AACCGGTTGACATTTTTTTTTATAATGGCC",
        "usr_label__primary": "demoP",
        "seq_annot__source_file": "/archive/demo.gb",
        "derived__product_kind": "selected_region",
        "seq_annot__features": [
            {
                "feature_id": "feat_promoter",
                "feature_order": 1,
                "feature_type": "misc_feature",
                "label": "pred. demoP",
                "role_hint": None,
                "start_0": 2,
                "end_0": 28,
                "strand": 1,
                "confidence": "high",
            },
            {
                "feature_id": "feat_m35",
                "feature_order": 2,
                "feature_type": "misc_feature",
                "label": "-35",
                "role_hint": "sigma70_minus35",
                "start_0": 6,
                "end_0": 12,
                "strand": 1,
                "confidence": "high",
            },
            {
                "feature_id": "feat_tfbs",
                "feature_order": 3,
                "feature_type": "misc_feature",
                "label": "LexA-",
                "role_hint": "TFBS",
                "start_0": 10,
                "end_0": 18,
                "strand": -1,
                "confidence": "high",
            },
            {
                "feature_id": "feat_m10",
                "feature_order": 4,
                "feature_type": "misc_feature",
                "label": "-10",
                "role_hint": "sigma70_minus10",
                "start_0": 20,
                "end_0": 26,
                "strand": 1,
                "confidence": "high",
            },
        ],
    }


def _genbank_row_with_obstructed_sigma_label() -> dict[str, object]:
    row = _genbank_row()
    row["sequence"] = "AACCGGTTGACATTTTTTTTTATAATGGCCAAAAAA"
    row["seq_annot__features"] = [
        {
            "feature_id": "feat_promoter",
            "feature_order": 1,
            "feature_type": "misc_feature",
            "label": "pred. demoP",
            "role_hint": None,
            "start_0": 2,
            "end_0": 34,
            "strand": 1,
            "confidence": "high",
        },
        {
            "feature_id": "feat_m35",
            "feature_order": 2,
            "feature_type": "misc_feature",
            "label": "-35",
            "role_hint": "sigma70_minus35",
            "start_0": 6,
            "end_0": 12,
            "strand": 1,
            "confidence": "high",
        },
        {
            "feature_id": "feat_obstruction",
            "feature_order": 3,
            "feature_type": "misc_feature",
            "label": "LexA-",
            "role_hint": "TFBS",
            "start_0": 8,
            "end_0": 18,
            "strand": 1,
            "confidence": "high",
        },
        {
            "feature_id": "feat_m10",
            "feature_order": 4,
            "feature_type": "misc_feature",
            "label": "-10",
            "role_hint": "sigma70_minus10",
            "start_0": 24,
            "end_0": 30,
            "strand": 1,
            "confidence": "high",
        },
    ]
    return row


def _genbank_row_with_modern_reference_labels() -> dict[str, object]:
    return {
        "id": "seq2",
        "sequence": "A" * 120,
        "usr_label__primary": "cpxPp",
        "seq_annot__source_file": "/archive/cpxPp.gb",
        "derived__product_kind": "selected_region",
        "seq_annot__features": [
            {
                "feature_id": "source_fragment",
                "feature_order": 1,
                "feature_type": "misc_feature",
                "label": "cpxPp (upstream of cpxP)",
                "role_hint": None,
                "start_0": 0,
                "end_0": 120,
                "strand": 1,
                "confidence": "high",
            },
            {
                "feature_id": "promoter_call",
                "feature_order": 2,
                "feature_type": "misc_feature",
                "label": "cpxPp",
                "role_hint": None,
                "start_0": 30,
                "end_0": 111,
                "strand": 1,
                "confidence": "high",
            },
            {
                "feature_id": "predicted_tfbs",
                "feature_order": 3,
                "feature_type": "misc_feature",
                "label": "AraC-arabinose pred. TFBS",
                "role_hint": None,
                "start_0": 35,
                "end_0": 53,
                "strand": -1,
                "confidence": "medium",
            },
            {
                "feature_id": "operator",
                "feature_order": 4,
                "feature_type": "misc_feature",
                "label": "araO1",
                "role_hint": None,
                "start_0": 60,
                "end_0": 81,
                "strand": -1,
                "confidence": "medium",
            },
        ],
    }


def _adapter_columns() -> dict[str, str]:
    return {
        "sequence": "sequence",
        "annotations": "seq_annot__features",
        "id": "id",
        "overlay_text": "usr_label__primary",
        "source_file": "seq_annot__source_file",
        "product_kind": "derived__product_kind",
    }


def test_usr_genbank_adapter_maps_annotation_roles_to_sequence_rows_features() -> None:
    record = adapt_record(
        _genbank_row(),
        adapter_kind="usr_genbank_annotations_v1",
        adapter_columns=_adapter_columns(),
        adapter_policies={"overlay_text_template": "{overlay_text}\n{id}"},
        alphabet="DNA",
    )

    assert record.id == "seq1"
    assert record.display.overlay_text == "demoP\nseq1"
    assert record.meta["adapter"] == "usr_genbank_annotations_v1"
    assert record.meta["source_file"] == "/archive/demo.gb"
    assert record.meta["product_kind"] == "selected_region"

    by_id = {feature.id: feature for feature in record.features}
    assert by_id["seq1:genbank:feat_m35"].label == "TTGACA"
    assert by_id["seq1:genbank:feat_m35"].tags == ("promoter:sigma70_core:upstream",)
    assert by_id["seq1:genbank:feat_m10"].label == "TATAAT"
    assert by_id["seq1:genbank:feat_m10"].tags == ("promoter:sigma70_core:downstream",)
    assert by_id["seq1:genbank:feat_tfbs"].tags == ("tf:LexA",)
    assert by_id["seq1:genbank:feat_tfbs"].span.strand == "rev"
    assert record.display.tag_labels["tf:LexA"] == "LexA sites"
    assert record.display.tag_labels["promoter:sigma70_core:upstream"] == "-35 site"
    assert record.display.tag_labels["promoter:sigma70_core:downstream"] == "-10 site"
    assert record.display.tag_labels["genbank:promoter_region"] == "Promoter region"
    assert any(effect.kind == "span_link" for effect in record.effects)


def test_usr_genbank_adapter_uses_specific_annotation_semantics_not_generic_feature_bucket() -> None:
    record = adapt_record(
        _genbank_row_with_modern_reference_labels(),
        adapter_kind="usr_genbank_annotations_v1",
        adapter_columns=_adapter_columns(),
        alphabet="DNA",
    )

    by_id = {feature.id: feature for feature in record.features}

    assert "seq2:genbank:source_fragment" not in by_id
    assert by_id["seq2:genbank:promoter_call"].tags == ("genbank:promoter_region",)
    assert by_id["seq2:genbank:predicted_tfbs"].tags == ("tf:AraC-arabinose",)
    assert by_id["seq2:genbank:operator"].tags == ("genbank:operator_site",)
    assert "genbank:genbank_feature" not in record.display.tag_labels
    assert "genbank:source_fragment" not in record.display.tag_labels
    assert record.display.tag_labels["genbank:promoter_region"] == "Promoter region"
    assert record.display.tag_labels["genbank:operator_site"] == "Operator site"
    assert record.display.tag_labels["tf:AraC-arabinose"] == "AraC-arabinose sites"


def test_usr_genbank_interval_annotations_are_placed_by_sequence_rows_layout() -> None:
    record = adapt_record(
        _genbank_row_with_modern_reference_labels(),
        adapter_kind="usr_genbank_annotations_v1",
        adapter_columns=_adapter_columns(),
        alphabet="DNA",
    )
    style = resolve_style(
        preset="presentation_default",
        overrides={
            "show_reverse_complement": False,
            "connectors": False,
            "legend": False,
            "font_size_seq": 18,
            "font_size_label": 18,
            "legend_font_size": 18,
            "uniform_display_font_size": True,
        },
    )

    layout = compute_layout(record, style)

    assert "seq2:genbank:source_fragment" not in layout.feature_boxes
    assert "seq2:genbank:promoter_call" in layout.feature_boxes
    assert "seq2:genbank:operator" in layout.feature_boxes


def test_usr_genbank_interval_annotations_render_as_filled_spans_with_near_labels() -> None:
    record = adapt_record(
        _genbank_row_with_modern_reference_labels(),
        adapter_kind="usr_genbank_annotations_v1",
        adapter_columns=_adapter_columns(),
        alphabet="DNA",
    )
    by_id = {feature.id: feature for feature in record.features}

    assert "seq2:genbank:source_fragment" not in by_id
    assert by_id["seq2:genbank:promoter_call"].label == "cpxPp"
    assert by_id["seq2:genbank:promoter_call"].attrs["shape"] == "rounded_rect"
    assert by_id["seq2:genbank:promoter_call"].attrs["source"] == "usr_genbank"

    style = resolve_style(
        preset="presentation_default",
        overrides={
            "show_reverse_complement": False,
            "connectors": False,
            "legend": False,
            "font_size_seq": 18,
            "font_size_label": 18,
            "legend_font_size": 18,
            "uniform_display_font_size": True,
        },
    )

    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=Palette(style.palette))
    try:
        axis = fig.axes[0]
        rendered_text = {text.get_text() for text in axis.texts}

        assert "cpxPp (upstream of cpxP)" not in rendered_text
        assert "cpxPp" in rendered_text
        assert "araO1" in rendered_text
        assert not axis.lines
    finally:
        plt.close(fig)


def test_usr_genbank_render_uses_shared_near_feature_annotation_labels() -> None:
    record = adapt_record(
        _genbank_row(),
        adapter_kind="usr_genbank_annotations_v1",
        adapter_columns=_adapter_columns(),
        alphabet="DNA",
    )
    style = resolve_style(
        preset="presentation_default",
        overrides={
            "show_reverse_complement": False,
            "connectors": False,
            "legend": False,
            "font_size_seq": 18,
            "font_size_label": 18,
            "legend_font_size": 18,
            "uniform_display_font_size": True,
        },
    )

    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=Palette(style.palette))
    try:
        axis = fig.axes[0]
        local_labels = {text.get_text() for text in axis.texts if float(text.get_zorder()) == 6.2}
        assert {"-35 site", "LexA sites", "-10 site"}.issubset(local_labels)
    finally:
        plt.close(fig)


def test_usr_genbank_reverse_strand_near_labels_are_placed_away_from_sequence_strand() -> None:
    record = adapt_record(
        _genbank_row(),
        adapter_kind="usr_genbank_annotations_v1",
        adapter_columns=_adapter_columns(),
        alphabet="DNA",
    )
    style = resolve_style(
        preset="presentation_default",
        overrides={
            "show_reverse_complement": True,
            "connectors": False,
            "legend": False,
            "font_size_seq": 18,
            "font_size_label": 18,
            "legend_font_size": 18,
            "uniform_display_font_size": True,
        },
    )
    layout = compute_layout(record, style)

    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=Palette(style.palette))
    try:
        axis = fig.axes[0]
        lex_a_label = next(text for text in axis.texts if text.get_text() == "LexA sites")
        feature_box = layout.feature_boxes["seq1:genbank:feat_tfbs"]

        assert float(lex_a_label.get_position()[1]) < float(feature_box[1])
    finally:
        plt.close(fig)


def test_usr_genbank_interval_fill_labels_use_display_font_size() -> None:
    record = adapt_record(
        _genbank_row_with_modern_reference_labels(),
        adapter_kind="usr_genbank_annotations_v1",
        adapter_columns=_adapter_columns(),
        alphabet="DNA",
    )
    style = resolve_style(
        preset="presentation_default",
        overrides={
            "show_reverse_complement": False,
            "connectors": False,
            "legend": False,
            "font_size_seq": 22,
            "font_size_label": 22,
            "legend_font_size": 11,
            "uniform_display_font_size": True,
        },
    )

    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=Palette(style.palette))
    try:
        axis = fig.axes[0]
        promoter_label = next(text for text in axis.texts if text.get_text() == "cpxPp")

        assert float(promoter_label.get_fontsize()) >= 22.0
    finally:
        plt.close(fig)


def test_usr_genbank_sigma_sites_share_span_link_lane() -> None:
    record = adapt_record(
        _genbank_row(),
        adapter_kind="usr_genbank_annotations_v1",
        adapter_columns=_adapter_columns(),
        alphabet="DNA",
    )
    style = resolve_style(
        preset="presentation_default",
        overrides={
            "show_reverse_complement": False,
            "connectors": False,
            "font_size_seq": 18,
            "font_size_label": 18,
            "legend_font_size": 18,
            "uniform_display_font_size": True,
        },
    )

    layout = compute_layout(record, style)
    by_display_label = {
        str(feature.attrs.get("display_label")): feature
        for feature in record.features
        if feature.attrs.get("display_label") in {"-35 site", "-10 site"}
    }
    span_effect = next(effect for effect in record.effects if effect.kind == "span_link")
    span_track = int(span_effect.render.get("track", 0))

    assert layout.feature_track_by_id[str(by_display_label["-35 site"].id)] == span_track
    assert layout.feature_track_by_id[str(by_display_label["-10 site"].id)] == span_track


def test_usr_genbank_sigma_site_text_labels_align_when_one_top_label_is_blocked() -> None:
    record = adapt_record(
        _genbank_row_with_obstructed_sigma_label(),
        adapter_kind="usr_genbank_annotations_v1",
        adapter_columns=_adapter_columns(),
        alphabet="DNA",
    )
    style = resolve_style(
        preset="presentation_default",
        overrides={
            "show_reverse_complement": False,
            "connectors": False,
            "legend": False,
            "font_size_seq": 18,
            "font_size_label": 18,
            "legend_font_size": 18,
            "uniform_display_font_size": True,
        },
    )

    initialize_runtime()
    fig = render_record(record, renderer_name="sequence_rows", style=style, palette=Palette(style.palette))
    try:
        axis = fig.axes[0]
        y_by_text = {
            text.get_text(): float(text.get_position()[1])
            for text in axis.texts
            if text.get_text() in {"-35 site", "-10 site", "12 bp"}
        }

        assert y_by_text["-35 site"] == pytest.approx(y_by_text["12 bp"])
        assert y_by_text["-10 site"] == pytest.approx(y_by_text["12 bp"])
    finally:
        plt.close(fig)


def test_sequence_rows_can_disable_content_radius_balance_to_tighten_footer_legend() -> None:
    record = adapt_record(
        _genbank_row_with_obstructed_sigma_label(),
        adapter_kind="usr_genbank_annotations_v1",
        adapter_columns=_adapter_columns(),
        alphabet="DNA",
    )

    def _bottom_strand_to_legend_gap(
        balance_content_radius: bool,
        *,
        legend_content_gap_px: float | None = None,
    ) -> float:
        overrides: dict[str, object] = {
            "show_reverse_complement": True,
            "connectors": False,
            "legend": True,
            "legend_mode": "bottom",
            "legend_height_px": 44.0,
            "legend_pad_px": 4.0,
            "font_size_seq": 18,
            "font_size_label": 18,
            "legend_font_size": 11,
            "uniform_display_font_size": True,
            "balance_content_radius": balance_content_radius,
        }
        if legend_content_gap_px is not None:
            overrides["legend_content_gap_px"] = legend_content_gap_px
        style = resolve_style(
            preset="presentation_default",
            overrides=overrides,
        )
        layout = compute_layout(record, style)
        initialize_runtime()
        fig = render_record(record, renderer_name="sequence_rows", style=style, palette=Palette(style.palette))
        try:
            axis = fig.axes[0]
            legend_ys = [float(text.get_position()[1]) for text in axis.texts if float(text.get_zorder()) == 10.0]
            assert legend_ys
            bottom_strand_y = float(layout.y_reverse - layout.sequence_extent_down)
            return bottom_strand_y - max(legend_ys)
        finally:
            plt.close(fig)

    balanced_gap = _bottom_strand_to_legend_gap(True)
    unbalanced_gap = _bottom_strand_to_legend_gap(False, legend_content_gap_px=12.0)

    assert unbalanced_gap < balanced_gap * 0.65
    assert unbalanced_gap < 70.0


def test_usr_genbank_adapter_rejects_missing_annotations_by_default() -> None:
    row = {
        "id": "seq1",
        "sequence": "ACGT",
        "seq_annot__features": None,
    }

    with pytest.raises(SchemaError, match="missing GenBank annotations"):
        adapt_record(
            row,
            adapter_kind="usr_genbank_annotations_v1",
            adapter_columns={"sequence": "sequence", "annotations": "seq_annot__features", "id": "id"},
            alphabet="DNA",
        )


def test_usr_genbank_adapter_rejects_non_integer_annotation_coordinates_as_schema_error() -> None:
    row = _genbank_row()
    features = list(row["seq_annot__features"])
    features[0] = {**features[0], "start_0": "left"}
    row["seq_annot__features"] = features

    with pytest.raises(SchemaError, match="GenBank annotation 0 start_0/end_0 must be integers"):
        adapt_record(
            row,
            adapter_kind="usr_genbank_annotations_v1",
            adapter_columns=_adapter_columns(),
            alphabet="DNA",
        )


def test_usr_genbank_adapter_rejects_out_of_bounds_annotation_coordinates_with_context() -> None:
    row = _genbank_row()
    features = list(row["seq_annot__features"])
    features[0] = {**features[0], "end_0": len(str(row["sequence"])) + 1}
    row["seq_annot__features"] = features

    with pytest.raises(SchemaError, match=r"GenBank annotation 0 span \[2, 31\) exceeds sequence length 30"):
        adapt_record(
            row,
            adapter_kind="usr_genbank_annotations_v1",
            adapter_columns=_adapter_columns(),
            alphabet="DNA",
        )


def test_usr_genbank_adapter_descriptor_and_render_contract_are_explicit() -> None:
    adapter = adapter_descriptor("usr_genbank_annotations_v1")
    contract = get_render_contract_descriptor("usr_genbank_annotation_render_v1")

    assert adapter.owner_tool == "usr"
    assert adapter.contract_kind == "usr_genbank_annotations_v1"
    assert adapter.supported_renderers == ("sequence_rows",)
    assert contract.accepted_renderers == ("sequence_rows",)


def test_usr_genbank_adapter_policy_normalizer_rejects_bad_min_per_record() -> None:
    adapter = adapter_descriptor("usr_genbank_annotations_v1")

    with pytest.raises(SchemaError, match="min_per_record must be int"):
        adapter.normalize_policies({"min_per_record": "many"}, "input.adapter.policies")


def test_usr_genbank_render_job_validates_with_use_case_contract(tmp_path: Path) -> None:
    input_path = write_parquet(tmp_path / "input.parquet", [_genbank_row()])
    payload = {
        "version": 3,
        "contract": {"kind": "usr_genbank_annotation_render_v1"},
        "results_root": str(tmp_path / "outputs"),
        "input": {
            "kind": "parquet",
            "path": str(input_path),
            "adapter": {
                "kind": "usr_genbank_annotations_v1",
                "columns": _adapter_columns(),
                "policies": {"require_non_empty": True},
            },
            "alphabet": "DNA",
        },
        "render": {"renderer": "sequence_rows", "style": {"preset": None, "overrides": {}}},
        "outputs": [{"kind": "images", "fmt": "png"}],
    }
    job_path = write_job(tmp_path / "job.yaml", payload)

    job = load_cruncher_showcase_job(job_path)

    assert job.contract.kind == "usr_genbank_annotation_render_v1"
    assert job.input.adapter.kind == "usr_genbank_annotations_v1"
