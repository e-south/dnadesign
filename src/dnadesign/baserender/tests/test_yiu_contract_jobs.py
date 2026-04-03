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

import dnadesign.baserender as baserender
from dnadesign.baserender.src.adapters.sequence_evidence_map_v1 import SequenceEvidenceMapV1Adapter

from .conftest import write_job


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


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
