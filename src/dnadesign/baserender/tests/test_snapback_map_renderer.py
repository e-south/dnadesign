"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_snapback_map_renderer.py

Direct renderer tests for snapback_map foldback semantics and layout behavior.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from dnadesign.baserender.src.adapters.snapback_visual_v1 import SnapbackVisualV1Adapter
from dnadesign.baserender.src.config import resolve_style
from dnadesign.baserender.src.render import Palette, render_record
from dnadesign.baserender.src.runtime import initialize_runtime


def _style_palette():
    initialize_runtime()
    style = resolve_style(preset=None, overrides={})
    return style, Palette(style.palette)


def _foldback_payload(*, complement_sequence: str = "GACTTGCAACT") -> dict[str, object]:
    return {
        "contract_kind": "snapback_visual_v1",
        "state_id": "demo.post_nick_foldback",
        "state_kind": "post_nick_foldback",
        "alphabet": "dna",
        "title": "post-nick foldback",
        "primary_sequence": "TCAGCATCTGA",
        "complement_sequence": complement_sequence,
        "primary_row_label": "Retained stem",
        "complement_row_label": "Foldback arm",
        "ligation_junction_boundary": 0,
        "protected_region_span": {"start": 0, "end": 4},
        "retained_stem_span": {"start": 0, "end": 4},
        "cap_span": {"start": 4, "end": 7},
        "foldback_revcomp_span": {"start": 7, "end": 11},
        "loop_geometry": {
            "kind": "hairpin_corner_triloop_v1",
            "source_cap_span": {"start": 4, "end": 6},
            "cap_extension_span": {"start": 6, "end": 7},
            "display_primary_span": {"start": 0, "end": 4},
            "display_complement_span": {"start": 7, "end": 11},
        },
        "pairings": [
            {"left_index": 0, "right_index": 10},
            {"left_index": 1, "right_index": 9},
            {"left_index": 2, "right_index": 8},
            {"left_index": 3, "right_index": 7},
        ],
        "primary_mismatch_positions": [],
        "complement_mismatch_positions": [],
        "meta": {"source": "test"},
    }


def _render_foldback(payload: dict[str, object]):
    style, palette = _style_palette()
    adapter = SnapbackVisualV1Adapter(columns={}, policies={}, alphabet="DNA")
    record = adapter.apply(payload, row_index=0)
    return render_record(record, renderer_name="snapback_map", style=style, palette=palette)


def test_foldback_renderer_uses_complement_sequence_and_offsets_terminals_from_loop() -> None:
    payload = _foldback_payload(complement_sequence="AACCGGTTCAA")
    fig = _render_foldback(payload)
    try:
        ax = fig.axes[0]
        texts_by_gid = {text.get_gid(): text for text in ax.texts if text.get_gid()}

        complement_indices = range(
            payload["foldback_revcomp_span"]["end"] - 1,
            payload["foldback_revcomp_span"]["start"] - 1,
            -1,
        )
        observed_foldback = [texts_by_gid[f"complement-base-{index}"].get_text() for index in complement_indices]
        expected_foldback = [payload["complement_sequence"][index] for index in complement_indices]

        assert observed_foldback == expected_foldback
        assert sum(1 for text in ax.texts if text.get_text() == "Origin") == 1
        assert all(text.get_text() != "Nick" for text in ax.texts)
        assert any(text.get_text() == "source cap" for text in ax.texts)
        assert any(text.get_text() == "extension" for text in ax.texts)
        assert any(text.get_text() == "protected overlap" for text in ax.texts)
        assert "foldback-loop-backbone" in {patch.get_gid() for patch in ax.patches if patch.get_gid()}

        cap_indices = range(payload["cap_span"]["start"], payload["cap_span"]["end"])
        max_cap_x = max(texts_by_gid[f"cap-base-{index}"].get_position()[0] for index in cap_indices)
        assert texts_by_gid["primary-end-terminal"].get_position()[0] > max_cap_x + 0.1
        assert texts_by_gid["complement-end-terminal"].get_position()[0] > max_cap_x + 0.1
        assert texts_by_gid["origin-label"].get_position()[0] < 0.0
        assert texts_by_gid["origin-label"].get_position()[1] > texts_by_gid["primary-start-terminal"].get_position()[1]
    finally:
        plt.close(fig)


def test_foldback_renderer_omits_extension_label_when_cap_extension_is_empty() -> None:
    payload = _foldback_payload()
    payload["loop_geometry"]["source_cap_span"] = {"start": 4, "end": 7}
    payload["loop_geometry"]["cap_extension_span"] = {"start": 7, "end": 7}

    fig = _render_foldback(payload)
    try:
        labels = {text.get_text() for text in fig.axes[0].texts}
        assert "source cap" in labels
        assert "extension" not in labels
    finally:
        plt.close(fig)


def test_foldback_renderer_omits_source_cap_label_when_source_cap_is_empty() -> None:
    payload = _foldback_payload()
    payload["loop_geometry"]["source_cap_span"] = {"start": 4, "end": 4}
    payload["loop_geometry"]["cap_extension_span"] = {"start": 4, "end": 7}

    fig = _render_foldback(payload)
    try:
        labels = {text.get_text() for text in fig.axes[0].texts}
        assert "source cap" not in labels
        assert "extension" in labels
    finally:
        plt.close(fig)
