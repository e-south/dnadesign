"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_yiu_contract_jobs.py

Tests for direct YIU render-contract rendering through the public baserender
job surface.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import dnadesign.baserender as baserender
from dnadesign.baserender.src.config import resolve_style
from dnadesign.baserender.src.core import Record
from dnadesign.baserender.src.render.layout import comp, compute_layout

from .conftest import write_job


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_run_job_renders_yiu_linear_state_contract_with_iupac_input(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "inputs" / "hairpin_pcr_linear_insert.json",
        {
            "contract_kind": "yiu_linear_state_v1",
            "state_id": "hairpin_pcr_linear_insert",
            "topology_kind": "linear_dsdna",
            "alphabet": "iupac_dna",
            "primary_sequence": "CCTCAGCCCGCTGATCCCTATCAGTGATAGAR",
            "complement_sequence": "YTCTATCACTGATAGGGATCAGCGGGCTGAGG",
            "segments": [
                {"segment_id": "snapback_seed", "source_start": 0, "source_end": 14, "state_start": 0, "state_end": 14},
                {
                    "segment_id": "assembled_payload",
                    "source_start": 14,
                    "source_end": 32,
                    "state_start": 14,
                    "state_end": 32,
                },
            ],
            "annotations": [],
            "cuts": [],
            "junctions": [{"id": "payload_assembly_junction", "join_index": 23}],
            "fragments": [],
            "display": {"title": "Split-payload insert"},
            "meta": {"evidence_mode": "pattern_compatibility"},
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
                "adapter": {"kind": "yiu_linear_state_v1"},
                "alphabet": "iupac_dna",
            },
            "render": {"renderer": "sequence_rows", "style": {"preset": None, "overrides": {}}},
            "outputs": [{"kind": "images", "path": "../renders/hairpin_pcr_linear_insert.pdf", "fmt": "pdf"}],
            "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
        },
    )

    report = baserender.run_job(job_path, caller_root=tmp_path)

    assert Path(report.outputs["images_path"]).exists()


def test_iupac_dna_sequence_rows_supports_reverse_complement_layout() -> None:
    assert comp("ACGTRYSWKMBDHVN") == "TGCAYRSWMKVHDBN"

    record = Record(
        id="iupac_linear_state",
        alphabet="IUPAC_DNA",
        sequence="ACGTRYSWKMBDHVN",
    ).validate()
    single_row = compute_layout(
        record,
        resolve_style(
            preset="presentation_default",
            overrides={"show_reverse_complement": False},
        ),
    )
    dual_row = compute_layout(
        record,
        resolve_style(
            preset="presentation_default",
            overrides={"show_reverse_complement": True},
        ),
    )

    assert dual_row.y_forward > dual_row.y_reverse
    assert dual_row.height > single_row.height


def test_run_job_renders_yiu_hairpin_topology_contract(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "inputs" / "ligated_ssdna_hairpin.json",
        {
            "contract_kind": "yiu_hairpin_topology_v1",
            "state_id": "ligated_ssdna_hairpin",
            "topology_kind": "ssdna_hairpin",
            "sequence": "CCTCAGCCCGCTGATCAGCGGGCTGAGG",
            "stem_left_span": {"start": 0, "end": 8},
            "stem_right_span": {"start": 20, "end": 28},
            "loop_span": {"start": 8, "end": 20},
            "pair_map": [
                {"left_index": 0, "right_index": 27},
                {"left_index": 1, "right_index": 26},
                {"left_index": 2, "right_index": 25},
            ],
            "adapter_branches": [],
            "annotations": [],
            "display": {"title": "Ligation hairpin"},
            "meta": {"evidence_mode": "concrete_realization"},
        },
    )
    job_path = write_job(
        tmp_path / "jobs" / "ligated_ssdna_hairpin.job.yaml",
        {
            "version": 3,
            "results_root": "..",
            "input": {
                "kind": "json",
                "path": "../inputs/ligated_ssdna_hairpin.json",
                "adapter": {"kind": "yiu_hairpin_topology_v1"},
                "alphabet": "DNA",
            },
            "render": {"renderer": "hairpin_cartoon", "style": {"preset": None, "overrides": {}}},
            "outputs": [{"kind": "images", "path": "../renders/ligated_ssdna_hairpin.pdf", "fmt": "pdf"}],
            "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
        },
    )

    report = baserender.run_job(job_path, caller_root=tmp_path)

    assert Path(report.outputs["images_path"]).exists()


def test_run_job_renders_yiu_topology_cartoon_contract(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "inputs" / "circularized_payload_candidate.json",
        {
            "contract_kind": "yiu_topology_cartoon_v1",
            "state_id": "circularized_payload_candidate",
            "topology_kind": "circular_duplex",
            "sequence": "CCGATGTCCCTATCAGTGATAGAGAGGGGGGGGGGGGCCTCAGCCCGCTGA",
            "segments": [
                {"segment_id": "payload_left", "source_start": 6, "source_end": 15, "state_start": 6, "state_end": 15},
                {
                    "segment_id": "payload_right",
                    "source_start": 21,
                    "source_end": 31,
                    "state_start": 15,
                    "state_end": 25,
                },
            ],
            "annotations": [],
            "cuts": [],
            "junctions": [{"id": "circularized_payload_junction", "join_index": 15}],
            "fragments": [],
            "display": {"title": "Circularized payload"},
            "meta": {"evidence_mode": "concrete_realization"},
        },
    )
    job_path = write_job(
        tmp_path / "jobs" / "circularized_payload_candidate.job.yaml",
        {
            "version": 3,
            "results_root": "..",
            "input": {
                "kind": "json",
                "path": "../inputs/circularized_payload_candidate.json",
                "adapter": {"kind": "yiu_topology_cartoon_v1"},
                "alphabet": "DNA",
            },
            "render": {"renderer": "topology_cartoon", "style": {"preset": None, "overrides": {}}},
            "outputs": [{"kind": "images", "path": "../renders/circularized_payload_candidate.pdf", "fmt": "pdf"}],
            "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
        },
    )

    report = baserender.run_job(job_path, caller_root=tmp_path)

    assert Path(report.outputs["images_path"]).exists()
