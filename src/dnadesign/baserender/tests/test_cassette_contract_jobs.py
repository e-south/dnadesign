"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_cassette_contract_jobs.py

Tests for direct JSON/JSONL cassette visual-contract rendering through the
public baserender job surface.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import dnadesign.baserender as baserender
from dnadesign.baserender.cli import app as baserender_app
from dnadesign.baserender.src.core import SchemaError

from .conftest import write_job

runner = CliRunner()


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _linear_duplex_payload(*, view_id: str, solution_id: str, title: str) -> dict[str, object]:
    return {
        "version": 1,
        "kind": "linear_duplex_v1",
        "view_id": view_id,
        "solution_id": solution_id,
        "title": title,
        "coordinate_semantics": "boundary_inclusive_v2",
        "primary_sequence_5to3": "TTTACCTCAGCAAAGCTGAGGTAAA",
        "sequence_span": {"start": 0, "end": 25},
        "cassette_span": {"start": 0, "end": 25},
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
            },
            {
                "id": "right_site",
                "variant_id": "Nt.BbvCI",
                "specificity_id": "BbvCI",
                "start": 16,
                "end": 23,
                "orientation": "reverse",
                "intent": "intended_right",
                "label": "Nt.BbvCI",
                "site_target_strand": "complement",
            },
        ],
        "nick_events": [
            {
                "id": "left_nick",
                "boundary": 7,
                "target_strand": "complement",
                "source_site_id": "left_site",
                "intent": "intended_left",
                "label": "Nick",
            },
            {
                "id": "right_nick",
                "boundary": 20,
                "target_strand": "complement",
                "source_site_id": "right_site",
                "intent": "intended_right",
                "label": "Nick",
            },
        ],
        "bounded_segment": {
            "start_boundary": 7,
            "end_boundary": 20,
            "target_strand": "complement",
            "label": "Bounded nicked segment",
        },
        "labels": [{"text": "Target strand: complement", "placement": "header"}],
        "meta": {
            "rank": 1,
            "left_variant_id": "Nb.BbvCI",
            "right_variant_id": "Nt.BbvCI",
            "left_boundary": 7,
            "right_boundary": 20,
            "bounded_length_nt": 13,
        },
    }


def _hairpin_payload() -> dict[str, object]:
    return {
        "version": 1,
        "kind": "ssdna_hairpin_v1",
        "view_id": "hit_001.ssdna_hairpin",
        "solution_id": "abc123def456",
        "title": "Hit 1 - ssDNA hairpin",
        "primary_sequence_5to3": "ACCTCAGCAAAGCTGAGGT",
        "topology": {
            "stem5p_span": {"start": 0, "end": 7},
            "loop_span": {"start": 7, "end": 12},
            "stem3p_span": {"start": 12, "end": 19},
        },
        "pair_map": [
            {"left_index": 0, "right_index": 18},
            {"left_index": 1, "right_index": 17},
            {"left_index": 2, "right_index": 16},
        ],
        "feature_spans": [
            {
                "id": "left_site_projection",
                "start": 1,
                "end": 7,
                "semantic": "motif_projection",
                "label": "Nb.BbvCI motif",
            },
            {
                "id": "right_site_projection",
                "start": 12,
                "end": 18,
                "semantic": "motif_projection",
                "label": "Nt.BbvCI motif",
            },
        ],
        "duplex_derived_annotations": [
            {
                "kind": "informational_note",
                "text": "Nicking is defined in the linear duplex interpretation.",
            }
        ],
        "meta": {"rank": 1, "left_variant_id": "Nb.BbvCI", "right_variant_id": "Nt.BbvCI"},
    }


def test_run_job_renders_linear_duplex_contract_from_json_path(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "inputs" / "linear_duplex.v1.json",
        _linear_duplex_payload(
            view_id="hit_001.linear_duplex",
            solution_id="abc123def456",
            title="Hit 1 - Linear duplex",
        ),
    )
    job_path = write_job(
        tmp_path / "jobs" / "linear_duplex.job.yaml",
        {
            "version": 3,
            "results_root": "..",
            "input": {
                "kind": "json",
                "path": "../inputs/linear_duplex.v1.json",
                "adapter": {"kind": "duplex_sequence_v1"},
                "alphabet": "DNA",
            },
            "render": {
                "renderer": "sequence_rows",
                "style": {
                    "preset": "cassette_duplex_qa",
                    "overrides": {"show_reverse_complement": True, "show_coordinate_ticks": True},
                },
            },
            "outputs": [{"kind": "images", "path": "../renders/linear_duplex.pdf", "fmt": "pdf"}],
            "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
        },
    )

    report = baserender.run_job(job_path, caller_root=tmp_path)

    assert Path(report.outputs["images_path"]) == (tmp_path / "renders" / "linear_duplex.pdf").resolve()
    assert Path(report.outputs["images_path"]).exists()


def test_run_job_renders_hairpin_contract_from_json_path(tmp_path: Path) -> None:
    _write_json(tmp_path / "inputs" / "ssdna_hairpin.v1.json", _hairpin_payload())
    job_path = write_job(
        tmp_path / "jobs" / "ssdna_hairpin.job.yaml",
        {
            "version": 3,
            "results_root": "..",
            "input": {
                "kind": "json",
                "path": "../inputs/ssdna_hairpin.v1.json",
                "adapter": {"kind": "hairpin_topology_v1"},
                "alphabet": "DNA",
            },
            "render": {
                "renderer": "hairpin_cartoon",
                "style": {
                    "preset": "cassette_hairpin_qa",
                    "overrides": {"show_pair_rungs": True, "show_loop_label": True},
                },
            },
            "outputs": [{"kind": "images", "path": "../renders/ssdna_hairpin.pdf", "fmt": "pdf"}],
            "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
        },
    )

    report = baserender.run_job(job_path, caller_root=tmp_path)

    assert Path(report.outputs["images_path"]) == (tmp_path / "renders" / "ssdna_hairpin.pdf").resolve()
    assert Path(report.outputs["images_path"]).exists()


def test_run_job_renders_duplex_contact_sheet_from_jsonl_path(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "inputs" / "top_hits.linear_duplex.v1.jsonl",
        [
            _linear_duplex_payload(
                view_id="hit_001.linear_duplex",
                solution_id="abc123def456",
                title="Hit 1 - Linear duplex",
            ),
            _linear_duplex_payload(
                view_id="hit_002.linear_duplex",
                solution_id="def456ghi789",
                title="Hit 2 - Linear duplex",
            ),
        ],
    )
    job_path = write_job(
        tmp_path / "jobs" / "top_hits_duplex.job.yaml",
        {
            "version": 3,
            "results_root": "..",
            "input": {
                "kind": "jsonl",
                "path": "../inputs/top_hits.linear_duplex.v1.jsonl",
                "adapter": {"kind": "duplex_sequence_v1"},
                "alphabet": "DNA",
            },
            "render": {
                "renderer": "sequence_rows",
                "style": {
                    "preset": "cassette_duplex_contact_sheet",
                    "overrides": {"show_reverse_complement": True, "show_coordinate_ticks": True},
                },
            },
            "outputs": [{"kind": "images", "path": "../renders/top_hits_duplex_qa_sheet.pdf", "fmt": "pdf"}],
            "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
        },
    )

    report = baserender.run_job(job_path, caller_root=tmp_path)

    assert Path(report.outputs["images_path"]) == (tmp_path / "renders" / "top_hits_duplex_qa_sheet.pdf").resolve()
    assert Path(report.outputs["images_path"]).exists()


def test_run_job_rejects_zero_length_bounded_segment_contract(tmp_path: Path) -> None:
    payload = _linear_duplex_payload(
        view_id="hit_001.linear_duplex",
        solution_id="abc123def456",
        title="Hit 1 - Linear duplex",
    )
    payload["bounded_segment"] = {
        "start_boundary": 7,
        "end_boundary": 7,
        "target_strand": "complement",
        "label": "Bounded nicked segment",
    }
    _write_json(
        tmp_path / "inputs" / "linear_duplex.v1.json",
        payload,
    )
    job_path = write_job(
        tmp_path / "jobs" / "linear_duplex.job.yaml",
        {
            "version": 3,
            "results_root": "..",
            "input": {
                "kind": "json",
                "path": "../inputs/linear_duplex.v1.json",
                "adapter": {"kind": "duplex_sequence_v1"},
                "alphabet": "DNA",
            },
            "render": {
                "renderer": "sequence_rows",
                "style": {"preset": "cassette_duplex_qa", "overrides": {}},
            },
            "outputs": [{"kind": "images", "path": "../renders/linear_duplex.pdf", "fmt": "pdf"}],
            "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
        },
    )

    with pytest.raises(SchemaError, match="bounded segment"):
        baserender.run_job(job_path, caller_root=tmp_path)


def test_run_job_rejects_out_of_range_nick_boundary_contract(tmp_path: Path) -> None:
    payload = _linear_duplex_payload(
        view_id="hit_001.linear_duplex",
        solution_id="abc123def456",
        title="Hit 1 - Linear duplex",
    )
    payload["nick_events"][0]["boundary"] = 26
    _write_json(
        tmp_path / "inputs" / "linear_duplex.v1.json",
        payload,
    )
    job_path = write_job(
        tmp_path / "jobs" / "linear_duplex.job.yaml",
        {
            "version": 3,
            "results_root": "..",
            "input": {
                "kind": "json",
                "path": "../inputs/linear_duplex.v1.json",
                "adapter": {"kind": "duplex_sequence_v1"},
                "alphabet": "DNA",
            },
            "render": {
                "renderer": "sequence_rows",
                "style": {"preset": "cassette_duplex_qa", "overrides": {}},
            },
            "outputs": [{"kind": "images", "path": "../renders/linear_duplex.pdf", "fmt": "pdf"}],
            "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
        },
    )

    with pytest.raises(SchemaError, match="nick boundary"):
        baserender.run_job(job_path, caller_root=tmp_path)


def test_run_job_rejects_missing_json_contract_input(tmp_path: Path) -> None:
    job_path = write_job(
        tmp_path / "jobs" / "missing.job.yaml",
        {
            "version": 3,
            "results_root": "..",
            "input": {
                "kind": "json",
                "path": "../inputs/missing_linear_duplex.v1.json",
                "adapter": {"kind": "duplex_sequence_v1"},
                "alphabet": "DNA",
            },
            "render": {
                "renderer": "sequence_rows",
                "style": {"preset": "cassette_duplex_qa", "overrides": {}},
            },
            "outputs": [{"kind": "images", "path": "../renders/linear_duplex.pdf", "fmt": "pdf"}],
            "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
        },
    )

    with pytest.raises(SchemaError, match="input.path does not exist"):
        baserender.run_job(job_path, caller_root=tmp_path)


def test_run_job_rejects_json_array_with_non_object_row(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "inputs" / "linear_duplex_rows.json",
        [
            _linear_duplex_payload(
                view_id="hit_001.linear_duplex",
                solution_id="abc123def456",
                title="Hit 1 - Linear duplex",
            ),
            "not-an-object",
        ],
    )
    job_path = write_job(
        tmp_path / "jobs" / "linear_duplex_rows.job.yaml",
        {
            "version": 3,
            "results_root": "..",
            "input": {
                "kind": "json",
                "path": "../inputs/linear_duplex_rows.json",
                "adapter": {"kind": "duplex_sequence_v1"},
                "alphabet": "DNA",
            },
            "render": {
                "renderer": "sequence_rows",
                "style": {"preset": "cassette_duplex_qa", "overrides": {}},
            },
            "outputs": [{"kind": "images", "path": "../renders/linear_duplex.pdf", "fmt": "pdf"}],
            "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
        },
    )

    with pytest.raises(SchemaError, match="JSON array item 1 must be an object"):
        baserender.run_job(job_path, caller_root=tmp_path)


def test_run_job_rejects_malformed_jsonl_contract_input(tmp_path: Path) -> None:
    path = tmp_path / "inputs" / "top_hits.linear_duplex.v1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    _linear_duplex_payload(
                        view_id="hit_001.linear_duplex",
                        solution_id="abc123def456",
                        title="Hit 1 - Linear duplex",
                    )
                ),
                "not-json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    job_path = write_job(
        tmp_path / "jobs" / "top_hits_duplex.job.yaml",
        {
            "version": 3,
            "results_root": "..",
            "input": {
                "kind": "jsonl",
                "path": "../inputs/top_hits.linear_duplex.v1.jsonl",
                "adapter": {"kind": "duplex_sequence_v1"},
                "alphabet": "DNA",
            },
            "render": {
                "renderer": "sequence_rows",
                "style": {"preset": "cassette_duplex_contact_sheet", "overrides": {}},
            },
            "outputs": [{"kind": "images", "path": "../renders/top_hits_duplex_qa_sheet.pdf", "fmt": "pdf"}],
            "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
        },
    )

    with pytest.raises(SchemaError, match="Could not parse JSONL line 2"):
        baserender.run_job(job_path, caller_root=tmp_path)


def test_cassette_job_normalize_cli_resolves_json_contract_paths(tmp_path: Path) -> None:
    run_root = tmp_path / "cassette_run"
    _write_json(
        run_root / "views" / "linear_duplex.v1.json",
        _linear_duplex_payload(
            view_id="hit_001.linear_duplex",
            solution_id="abc123def456",
            title="Hit 1 - Linear duplex",
        ),
    )
    job_path = write_job(
        run_root / "baserender_jobs" / "linear_duplex.job.yaml",
        {
            "version": 3,
            "results_root": "..",
            "input": {
                "kind": "json",
                "path": "../views/linear_duplex.v1.json",
                "adapter": {"kind": "duplex_sequence_v1"},
                "alphabet": "DNA",
            },
            "render": {
                "renderer": "sequence_rows",
                "style": {"preset": "cassette_duplex_qa", "overrides": {}},
            },
            "outputs": [{"kind": "images", "path": "../renders/linear_duplex.pdf", "fmt": "pdf"}],
            "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
        },
    )
    normalized_path = tmp_path / "normalized" / "linear_duplex.job.yaml"

    result = runner.invoke(
        baserender_app,
        ["job", "normalize", str(job_path), "--out", str(normalized_path)],
        color=False,
    )

    assert result.exit_code == 0
    normalized = normalized_path.read_text(encoding="utf-8")
    assert str((run_root / "views" / "linear_duplex.v1.json").resolve()) in normalized
