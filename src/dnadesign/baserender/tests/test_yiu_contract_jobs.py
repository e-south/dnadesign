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

import dnadesign.baserender as baserender

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
