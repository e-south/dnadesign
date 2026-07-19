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

import dnadesign.baserender as baserender
from dnadesign.baserender.src.public import api as public_api
from dnadesign.baserender.src.public.sequence_panel_layout import normalize_panel_image


def _densegen_row() -> dict[str, object]:
    return {
        "id": "r1",
        "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
            {"regulator": "cpxR", "orientation": "fwd", "sequence": "TATAAT", "offset": 23},
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
    config = baserender.sequence_panel_config_for_adapter("densegen_tfbs")
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
