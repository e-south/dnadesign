"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_sequence_panel_layout.py

Regression tests for the public sequence-panel title and pixel-layout contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import FancyBboxPatch

import dnadesign.baserender as baserender
from dnadesign.baserender.src.public import api as public_api
from dnadesign.baserender.src.public.sequence_panel_layout import normalize_panel_image

_PROMOTER_PANEL_PROFILE = "promoter_compact_slide.v1"


def _densegen_row() -> dict[str, object]:
    return {
        "id": "r1",
        "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
            {"regulator": "cpxR", "orientation": "fwd", "sequence": "TATAAT", "offset": 23},
        ],
    }


def _crowded_densegen_promoter_row() -> dict[str, object]:
    return {
        "id": "a52819bb39e768a258df0a790ee8a27241450490",
        "sequence": "TAGACACCTGTGTACATCCACAATATAATACTGGGTTGGGTCTAGGTCAACATCTCTGTC",
        "densegen__used_tfbs_detail": [
            {
                "part_kind": "tfbs",
                "regulator": "lexA_CTGTATAWAWWHACA",
                "orientation": "fwd",
                "sequence": "ACCTGTGTACATCCACAAT",
                "offset": 5,
                "offset_raw": 5,
            },
            {
                "part_kind": "tfbs",
                "regulator": "background",
                "orientation": "fwd",
                "sequence": "TACTGGGTTGGGTCTA",
                "offset": 28,
                "offset_raw": 28,
            },
            {
                "part_kind": "tfbs",
                "regulator": "background",
                "orientation": "rev",
                "sequence": "GACAGAGATGTTGACCT",
                "offset": 43,
                "offset_raw": 43,
            },
            {
                "part_kind": "fixed_element",
                "role": "upstream",
                "constraint_name": "sigma70_core",
                "sequence": "TAGACA",
                "variant_id": "e",
                "spacer_length": 17,
                "placement_index": 0,
                "offset": 0,
                "offset_raw": 0,
            },
            {
                "part_kind": "fixed_element",
                "role": "downstream",
                "constraint_name": "sigma70_core",
                "sequence": "TATAAT",
                "variant_id": "consensus",
                "spacer_length": 17,
                "placement_index": 0,
                "offset": 23,
                "offset_raw": 23,
            },
        ],
    }


def _capture_figure_text(monkeypatch) -> list[str]:
    captured_text: list[str] = []
    original_figure_rgba = public_api._figure_rgba

    def _capture(fig):
        captured_text.extend(text.get_text() for axis in fig.axes for text in axis.texts)
        return original_figure_rgba(fig)

    monkeypatch.setattr("dnadesign.baserender.src.public.api._figure_rgba", _capture)
    return captured_text


def test_public_sequence_panel_contract_renders_caller_title_inside_panel(monkeypatch) -> None:
    title = "Candidate ES42 · ethanol view · rank 3"
    captured_text = _capture_figure_text(monkeypatch)

    result = baserender.render_sequence_panel_image(
        _densegen_row(),
        adapter_kind="densegen_tfbs",
        style_profile=_PROMOTER_PANEL_PROFILE,
        target_width_px=420,
        target_height_px=140,
        title=title,
    )

    assert result.diagnostics.title == title
    assert result.diagnostics.record_label is None
    assert title in captured_text


def test_public_sequence_panel_title_preserves_adapter_record_label(monkeypatch) -> None:
    row = {
        "id": "seq1",
        "sequence": "AACCGGTTGACATTTTTTTTTATAATGGCC",
        "usr_label__primary": "demoP",
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
            }
        ],
    }
    title = "Candidate native control · rank 1"
    captured_text = _capture_figure_text(monkeypatch)

    result = baserender.render_sequence_panel_image(
        row,
        adapter_kind="usr_genbank_annotations_v1",
        style_profile=_PROMOTER_PANEL_PROFILE,
        target_width_px=420,
        target_height_px=140,
        title=title,
    )

    assert result.diagnostics.title == title
    assert result.diagnostics.record_label == "demoP"
    assert f"{title}\ndemoP" in captured_text


def test_sequence_panel_normalization_accepts_anchor_at_lower_image_boundary() -> None:
    source = np.full((20, 30, 4), 255, dtype=np.uint8)
    source[2:8, 4:26, :3] = 20

    normalized, anchor_y = normalize_panel_image(
        source,
        target_width_px=60,
        target_height_px=40,
        vertical_anchor="center",
        canvas_top_pad_px=0,
        source_anchor_y_px=20.0,
    )

    assert normalized.shape == (40, 60, 4)
    assert anchor_y == pytest.approx(20.0, abs=1.0)


def test_sequence_panel_profile_keeps_title_and_legend_near_sequence_content() -> None:
    title = "Candidate ES42 · ethanol view · rank 3"
    config = baserender.sequence_panel_config_for_adapter(
        "densegen_tfbs",
        style_profile=_PROMOTER_PANEL_PROFILE,
    )
    record = baserender.adapt_record(
        _densegen_row(),
        adapter_kind=config.adapter_kind,
        adapter_columns=config.adapter_columns,
        adapter_policies=config.adapter_policies,
    )
    record = replace(record, display=replace(record.display, overlay_text=title))
    fig = baserender.render_record_figure(
        record,
        renderer_name=config.renderer_name,
        style_preset=config.style_preset,
        style_overrides=config.style_overrides,
    )
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        renderer = fig.canvas.get_renderer()
        title_artist = next(text for text in ax.texts if text.get_text() == title)
        legend_artists = [text for text in ax.texts if text.get_text() in {"LexA sites", "CpxR sites"}]
        assert len(legend_artists) == 2

        feature_tops = [patch.get_window_extent(renderer=renderer).y1 for patch in ax.patches]
        sequence_patches = [patch for patch in ax.patches if str(patch.get_gid() or "").startswith("sequence:")]
        content_top = max(
            [*feature_tops, *(patch.get_window_extent(renderer=renderer).y1 for patch in sequence_patches)]
        )
        content_bottom = min(patch.get_window_extent(renderer=renderer).y0 for patch in sequence_patches)
        title_gap = title_artist.get_window_extent(renderer=renderer).y0 - content_top
        legend_top = max(text.get_window_extent(renderer=renderer).y1 for text in legend_artists)
        legend_gap = content_bottom - legend_top

        assert 4.0 <= title_gap <= 32.0
        assert 4.0 <= legend_gap <= 32.0
        assert title_artist.get_fontsize() == pytest.approx(24.0)
        assert {text.get_fontsize() for text in legend_artists} == {24.0}
    finally:
        plt.close(fig)


def test_sequence_panel_profile_routes_both_fixed_element_annotations_around_centered_title() -> None:
    title = "a52819bb39e768a258df0a790ee8a27241450490 · ciprofloxacin view · rank 1"
    config = baserender.sequence_panel_config_for_adapter(
        "densegen_tfbs",
        style_profile=_PROMOTER_PANEL_PROFILE,
    )
    record = baserender.adapt_record(
        _crowded_densegen_promoter_row(),
        adapter_kind=config.adapter_kind,
        adapter_columns=config.adapter_columns,
        adapter_policies=config.adapter_policies,
    )
    record = replace(record, display=replace(record.display, overlay_text=title))
    fig = baserender.render_record_figure(
        record,
        renderer_name=config.renderer_name,
        style_preset=config.style_preset,
        style_overrides=config.style_overrides,
    )
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        renderer = fig.canvas.get_renderer()
        title_artist = next(text for text in ax.texts if text.get_text() == title)
        title_box = title_artist.get_window_extent(renderer=renderer)
        annotation_artists = [
            text
            for text in ax.texts
            if text.get_text().startswith("-35 site (") or text.get_text().startswith("-10 site (")
        ]

        assert len(annotation_artists) == 2
        assert str(title_artist.get_ha()).lower() == "center"
        assert float(title_artist.get_position()[0]) == pytest.approx(sum(ax.get_xlim()) / 2.0)
        for annotation in annotation_artists:
            annotation_box = annotation.get_window_extent(renderer=renderer)
            assert not (
                max(title_box.x0, annotation_box.x0) < min(title_box.x1, annotation_box.x1)
                and max(title_box.y0, annotation_box.y0) < min(title_box.y1, annotation_box.y1)
            )
            assert annotation.get_fontsize() == pytest.approx(title_artist.get_fontsize())
            feature_boxes = [
                patch.get_window_extent(renderer=renderer)
                for patch in ax.patches
                if isinstance(patch, FancyBboxPatch)
                and patch.get_window_extent(renderer=renderer).height >= annotation_box.height * 1.25
            ]
            for feature_box in feature_boxes:
                overlaps_vertically = max(feature_box.y0, annotation_box.y0) < min(feature_box.y1, annotation_box.y1)
                if not overlaps_vertically:
                    continue
                horizontal_gap = max(feature_box.x0 - annotation_box.x1, annotation_box.x0 - feature_box.x1)
                assert horizontal_gap >= 8.0
    finally:
        plt.close(fig)
