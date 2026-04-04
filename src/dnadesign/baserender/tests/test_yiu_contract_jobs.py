"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_yiu_contract_jobs.py

Tests for direct YIU evidence-contract rendering through the public baserender
job surface.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pytest

import dnadesign.baserender as baserender
from dnadesign.baserender.src.adapters.sequence_evidence_map_v1 import SequenceEvidenceMapV1Adapter
from dnadesign.baserender.src.adapters.yiu_payload_visual_v1 import YiuPayloadVisualV1Adapter
from dnadesign.baserender.src.config import resolve_style
from dnadesign.baserender.src.render.effects.motif_logo import compute_motif_logo_geometry
from dnadesign.baserender.src.render.layout import compute_layout

from .conftest import write_job


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _canonical_tetr_pwm_rows() -> list[list[float]]:
    path = (
        Path(__file__).resolve().parents[2]
        / "cruncher"
        / "workspaces"
        / "demo_monotypic_tetr"
        / ".cruncher"
        / "demo_monotypic_tetr"
        / "normalized"
        / "motifs"
        / "westmann_tetr_mitomi"
        / "tetR.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))["matrix"]


def test_run_job_renders_sequence_evidence_map_contract(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "inputs" / "hairpin_pcr_linear_insert.json",
        {
            "contract_kind": "sequence_evidence_map_v1",
            "state_id": "hairpin_pcr_linear_insert",
            "topology_kind": "linear_dsdna",
            "alphabet": "iupac_dna",
            "primary_sequence": "TAGGGAAGGTCTCACACCTATAGAGCCTCAGCCCGCTGAATAGAG",
            "complement_sequence": "CTCTATT CAGCGGGCTGAGGCTCTATAGGTGTGAGACCTTCCCTA".replace(" ", ""),
            "owners": [
                {
                    "owner_id": "hairpin_pcr_forward_binding_region",
                    "row_id": "primary",
                    "start": 0,
                    "end": 6,
                    "display_label": "HP PCR Fwd",
                    "short_label": "HPF",
                },
                {
                    "owner_id": "retained_region",
                    "row_id": "primary",
                    "start": 6,
                    "end": 39,
                    "display_label": "Retained region",
                    "short_label": "RET",
                },
                {
                    "owner_id": "hairpin_pcr_reverse_binding_region",
                    "row_id": "primary",
                    "start": 39,
                    "end": 45,
                    "display_label": "HP PCR Rev",
                    "short_label": "HPR",
                },
                {
                    "owner_id": "retained_region",
                    "row_id": "complement",
                    "start": 0,
                    "end": 45,
                    "display_label": "Retained region",
                    "short_label": "RET",
                },
            ],
            "effect_tags": [
                {
                    "tag_id": "left_overhang",
                    "tag_kind": "payload_overhang_left",
                    "row_id": "primary",
                    "start": 6,
                    "end": 10,
                    "display_label": "Payload overhang L",
                    "short_label": "OvL",
                },
                {
                    "tag_id": "right_overhang",
                    "tag_kind": "payload_overhang_right",
                    "row_id": "primary",
                    "start": 18,
                    "end": 22,
                    "display_label": "Payload overhang R",
                    "short_label": "OvR",
                },
            ],
            "boundaries": [
                {
                    "boundary_id": "ligation_join",
                    "row_id": "primary",
                    "boundary": 18,
                    "boundary_kind": "ligation_junction",
                    "display_label": "Ligation",
                    "short_label": "Lig",
                }
            ],
            "pairings": [
                {
                    "pairing_id": "payload_pairing",
                    "primary_start": 6,
                    "primary_end": 10,
                    "complement_start": 39,
                    "complement_end": 43,
                    "display_label": "WC pairing",
                    "short_label": "WC",
                }
            ],
            "display": {"title": "Hairpin PCR insert"},
            "meta": {"evidence_mode": "nucleotide_truth"},
        },
    )
    job_path = write_job(
        tmp_path / "jobs" / "hairpin_pcr_linear_insert.job.yaml",
        {
            "version": 3,
            "results_root": "..",
            "input": {
                "kind": "json",
                "path": "../inputs/hairpin_pcr_linear_insert.json",
                "adapter": {"kind": "sequence_evidence_map_v1"},
                "alphabet": "iupac_dna",
            },
            "render": {"renderer": "nucleotide_evidence_map", "style": {"preset": None, "overrides": {}}},
            "outputs": [{"kind": "images", "path": "../renders/hairpin_pcr_linear_insert.pdf", "fmt": "pdf"}],
            "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
        },
    )

    report = baserender.run_job(job_path, caller_root=tmp_path)

    assert Path(report.outputs["images_path"]).exists()


def test_render_uses_explicit_complement_sequence_and_highlights_mismatch_bases() -> None:
    adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="IUPAC_DNA")
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
                "dim_base_indices": {
                    "primary": [0, 1, 2, 3, 4, 5, 6],
                    "complement": [0, 1, 2, 3, 4, 5, 6],
                },
                "base_highlights": {
                    "primary": [10],
                    "complement": [10],
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

    assert sum(1 for effect in record.effects if effect.kind == "boundary_marker") == 0

    fig = baserender.render(record, renderer="nucleotide_evidence_map", style={"preset": "presentation_default"})
    try:
        patch_by_gid = {patch.get_gid(): patch for patch in fig.axes[0].patches if patch.get_gid()}
        gids = set(patch_by_gid)
        labels = {text.get_text() for text in fig.axes[0].texts}
        line_segments = [(tuple(line.get_xdata()), tuple(line.get_ydata())) for line in fig.axes[0].lines]
    finally:
        plt.close(fig)

    assert "sequence:fwd:10:G:highlight" in gids
    assert "sequence:rev:10:A:highlight" in gids
    assert {"Left", "Right"}.issubset(labels)
    assert any(y0 == y1 and x0 != x1 for (x0, x1), (y0, y1) in line_segments)
    assert sum(1 for (x0, x1), (y0, y1) in line_segments if x0 != x1 and y0 != y1) >= 2
    dimmed_rgb = mcolors.to_rgb(patch_by_gid["sequence:fwd:0:C"].get_facecolor())
    payload_rgb = mcolors.to_rgb(patch_by_gid["sequence:fwd:8:C"].get_facecolor())
    assert sum(dimmed_rgb) > sum(payload_rgb)


def test_yiu_payload_visual_adapter_renders_pwm_layers_without_label_span_errors() -> None:
    adapter = YiuPayloadVisualV1Adapter(columns={}, policies={}, alphabet="IUPAC_DNA")
    record = adapter.apply(
        {
            "contract_kind": "yiu_payload_visual_v1",
            "state_id": "payload",
            "alphabet": "iupac_dna",
            "reference_payload_sequence": "CTCTATATCTGATATAGAG",
            "selected_payload_sequence": "CTCTATATCTGATATAGAG",
            "selected_complement_sequence": "GAGATATAGTGTATATCTC",
            "show_reference_payload_row": False,
            "junction": {"start": 8, "end": 12, "offsets": [0, 1, 2, 3]},
            "mismatches": [
                {
                    "payload_index": 9,
                    "junction_offset": 1,
                    "mutated_strand": "complement",
                    "native_base": "A",
                    "mutated_base": "T",
                    "opposing_base": "T",
                },
                {
                    "payload_index": 10,
                    "junction_offset": 2,
                    "mutated_strand": "complement",
                    "native_base": "C",
                    "mutated_base": "G",
                    "opposing_base": "G",
                },
            ],
            "motif_layers": [
                {
                    "motif_instance_id": "tetR_payload_site",
                    "tf_name": "tetR",
                    "motif_name": "tetr_demo",
                    "reference_strand": "+",
                    "start": 0,
                    "end": 17,
                    "label": "tetR (+)",
                    "matrix": _canonical_tetr_pwm_rows(),
                }
            ],
            "display": {"title": "TetR payload"},
            "meta": {"row_labels": {"primary": "Selected payload", "complement": "Selected complement"}},
        },
        row_index=0,
    )

    motif_feature = next(feature for feature in record.features if feature.id == "motif:tetR_payload_site")
    motif_effect = next(
        effect for effect in record.effects if effect.target.get("feature_id") == "motif:tetR_payload_site"
    )
    boundary_effects = [effect for effect in record.effects if effect.kind == "boundary_marker"]
    assert [effect.params["label"] for effect in boundary_effects] == ["", ""]
    assert motif_feature.label == "CTCTATATCTGATATAG"
    assert motif_feature.tags[0] == "motif:tetR_payload_site"
    assert motif_feature.attrs["style_token"] == "motif:tetR_payload_site"
    assert motif_feature.attrs["display_label"] == "tetR (+)"
    assert motif_effect.params["render_span"] == {"start": 0, "end": 19}
    assert motif_effect.params["observed_sequence_5to3"] == "CTCTATATCTGATATAGAG"
    assert len(motif_effect.params["matrix"]) == 19
    for got, expected in zip(motif_effect.params["matrix"][:17], _canonical_tetr_pwm_rows(), strict=True):
        assert got == pytest.approx(expected)
    assert motif_effect.params["matrix"][17] == [0.25, 0.25, 0.25, 0.25]
    assert motif_effect.params["matrix"][18] == [0.25, 0.25, 0.25, 0.25]

    fig = baserender.render(record, renderer="nucleotide_evidence_map", style={"preset": "presentation_default"})
    try:
        gids = {patch.get_gid() for patch in fig.axes[0].patches if patch.get_gid()}
    finally:
        plt.close(fig)

    assert any(gid.startswith("motif_logo:motif:tetR_payload_site:") for gid in gids)


def test_yiu_payload_visual_adapter_pwm_letters_follow_feature_fill_with_gray_deemphasis() -> None:
    adapter = YiuPayloadVisualV1Adapter(columns={}, policies={}, alphabet="IUPAC_DNA")
    record = adapter.apply(
        {
            "contract_kind": "yiu_payload_visual_v1",
            "state_id": "payload",
            "alphabet": "iupac_dna",
            "reference_payload_sequence": "CTCTATATCTGATATAGAG",
            "selected_payload_sequence": "CTCTATATCTGATATAGAG",
            "selected_complement_sequence": "GAGATATAGTGTATATCTC",
            "show_reference_payload_row": False,
            "junction": {"start": 8, "end": 12, "offsets": [0, 1, 2, 3]},
            "mismatches": [],
            "motif_layers": [
                {
                    "motif_instance_id": "tetR_payload_site",
                    "tf_name": "tetR",
                    "motif_name": "tetr_demo",
                    "reference_strand": "+",
                    "start": 0,
                    "end": 17,
                    "label": "tetR (+)",
                    "matrix": _canonical_tetr_pwm_rows(),
                }
            ],
            "display": {"title": "TetR payload"},
            "meta": {"row_labels": {"primary": "Selected payload", "complement": "Selected complement"}},
        },
        row_index=0,
    )

    fig = baserender.render(
        record,
        renderer="nucleotide_evidence_map",
        style={
            "preset": "presentation_default",
            "overrides": {
                "connectors": True,
                "palette": {"motif:tetR_payload_site": "#D68AA7"},
                "motif_logo": {
                    "letter_coloring": {
                        "mode": "match_window_seq",
                        "other_color": "#D1D5DB",
                        "observed_color_source": "feature_fill",
                    }
                },
            },
        },
    )
    try:
        patch_by_gid = {patch.get_gid(): patch for patch in fig.axes[0].patches if patch.get_gid()}
    finally:
        plt.close(fig)

    observed = mcolors.to_hex(
        patch_by_gid["motif_logo:motif:tetR_payload_site:0:C"].get_facecolor(),
        keep_alpha=False,
    )
    deemphasized = mcolors.to_hex(
        patch_by_gid["motif_logo:motif:tetR_payload_site:0:A"].get_facecolor(),
        keep_alpha=False,
    )
    assert observed == mcolors.to_hex("#D68AA7")
    assert deemphasized == mcolors.to_hex("#D1D5DB")


def test_yiu_payload_visual_adapter_preserves_reverse_strand_payload_wide_pwm_alignment() -> None:
    adapter = YiuPayloadVisualV1Adapter(columns={}, policies={}, alphabet="IUPAC_DNA")
    record = adapter.apply(
        {
            "contract_kind": "yiu_payload_visual_v1",
            "state_id": "payload",
            "alphabet": "iupac_dna",
            "reference_payload_sequence": "AACCGG",
            "selected_payload_sequence": "AACCGG",
            "selected_complement_sequence": "TTGGCC",
            "show_reference_payload_row": False,
            "junction": {"start": 1, "end": 5, "offsets": [0, 1, 2, 3]},
            "mismatches": [],
            "motif_layers": [
                {
                    "motif_instance_id": "minus_site",
                    "tf_name": "TF_MINUS",
                    "motif_name": "minus_demo",
                    "reference_strand": "-",
                    "start": 1,
                    "end": 4,
                    "label": "minus (-)",
                    "matrix": [
                        [0.97, 0.01, 0.01, 0.01],
                        [0.01, 0.97, 0.01, 0.01],
                        [0.01, 0.01, 0.97, 0.01],
                    ],
                }
            ],
            "display": {"title": "Reverse payload"},
            "meta": {"row_labels": {"primary": "Selected payload", "complement": "Selected complement"}},
        },
        row_index=0,
    )

    style = resolve_style(preset="presentation_default", overrides=None)
    layout = compute_layout(record, style)
    effect_index = next(index for index, effect in enumerate(record.effects) if effect.kind == "motif_logo")
    geometry = compute_motif_logo_geometry(record=record, effect_index=effect_index, layout=layout, style=style)

    assert geometry.render_start == 0
    assert geometry.render_end == 6
    assert geometry.observed == "TTGGCC"
    assert geometry.matrix[0] == pytest.approx((0.25, 0.25, 0.25, 0.25))
    assert geometry.matrix[1] == pytest.approx((0.01, 0.01, 0.97, 0.01))
    assert geometry.matrix[2] == pytest.approx((0.01, 0.97, 0.01, 0.01))
    assert geometry.matrix[3] == pytest.approx((0.97, 0.01, 0.01, 0.01))


def test_yiu_payload_visual_adapter_stacks_overlapping_same_strand_motifs_without_fixed_track_collisions() -> None:
    adapter = YiuPayloadVisualV1Adapter(columns={}, policies={}, alphabet="IUPAC_DNA")
    record = adapter.apply(
        {
            "contract_kind": "yiu_payload_visual_v1",
            "state_id": "payload",
            "alphabet": "iupac_dna",
            "reference_payload_sequence": "TTTTTCCCCCAAAA",
            "selected_payload_sequence": "TTTTTCCCCCAAAA",
            "selected_complement_sequence": "AAAAGGGGGGTTTT",
            "show_reference_payload_row": False,
            "junction": {"start": 5, "end": 9, "offsets": [0, 1, 2, 3]},
            "mismatches": [],
            "motif_layers": [
                {
                    "motif_instance_id": "baeR:0:11:+:1",
                    "tf_name": "baeR",
                    "motif_name": "baeR",
                    "reference_strand": "+",
                    "start": 0,
                    "end": 11,
                    "label": "baeR#1 (+)",
                    "matrix": [[0.97, 0.01, 0.01, 0.01]] * 11,
                },
                {
                    "motif_instance_id": "baeR:2:13:+:2",
                    "tf_name": "baeR",
                    "motif_name": "baeR",
                    "reference_strand": "+",
                    "start": 2,
                    "end": 13,
                    "label": "baeR#2 (+)",
                    "matrix": [[0.97, 0.01, 0.01, 0.01]] * 11,
                },
                {
                    "motif_instance_id": "baeR:3:14:+:3",
                    "tf_name": "baeR",
                    "motif_name": "baeR",
                    "reference_strand": "+",
                    "start": 3,
                    "end": 14,
                    "label": "baeR#3 (+)",
                    "matrix": [[0.97, 0.01, 0.01, 0.01]] * 11,
                },
            ],
            "display": {"title": "BaeR payload"},
            "meta": {"row_labels": {"primary": "Selected payload", "complement": "Selected complement"}},
        },
        row_index=0,
    )

    style = resolve_style(preset="presentation_default", overrides=None)
    layout = compute_layout(record, style)

    feature_tracks = {
        feature.id: layout.feature_track_by_id[feature.id]
        for feature in record.features
        if feature.id and feature.id.startswith("motif:")
    }
    assert set(feature_tracks.values()) == {0, 1, 2}
    assert {
        feature.attrs["style_token"] for feature in record.features if feature.id and feature.id.startswith("motif:")
    } == {
        "motif:baeR:0:11:+:1",
        "motif:baeR:2:13:+:2",
        "motif:baeR:3:14:+:3",
    }

    effect_indices = [idx for idx, effect in enumerate(record.effects) if effect.kind == "motif_logo"]
    effect_lanes = {layout.motif_logo_lane_by_effect[idx] for idx in effect_indices}
    assert effect_lanes == {0, 1, 2}
