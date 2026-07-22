"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/cassette/test_visual_publication.py

End-to-end publication tests for cassette view contracts and baserender jobs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import dnadesign.baserender as baserender
from dnadesign.cruncher.app.cassette_solve_workflow import run_cassette_solve
from dnadesign.cruncher.app.cassette_workflow import run_cassette_design


def _write_cassette_spec(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspaces" / "demo_cassette"
    spec_path = workspace / "configs" / "cassettes" / "demo_hairpin.cassette.yaml"
    catalog_path = workspace / "inputs" / "nickases" / "demo.nickases.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "nb_left",
                            "recognition_sequence": "AACGA",
                            "nicked_site_strand": "forward",
                            "cut_offset": 2,
                        },
                        {
                            "id": "nb_right",
                            "recognition_sequence": "AACGA",
                            "nicked_site_strand": "reverse",
                            "cut_offset": 3,
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    spec_path.write_text(
        yaml.safe_dump(
            {
                "cassette": {
                    "schema_version": 2,
                    "name": "demo_hairpin",
                    "topology": {
                        "stem5p_arm": "AACGAT",
                        "loop": "TT",
                        "stem3p_arm_mode": "derived_reverse_complement",
                    },
                    "construct_context": {"left_flank": "", "right_flank": ""},
                    "nicking": {
                        "target_strand": "primary",
                        "left": {"nickase": "nb_left", "nick_window": {"start": 0, "end": 3}},
                        "right": {"nickase": "nb_right", "nick_window": {"start": 11, "end": 13}},
                    },
                    "catalog": {"path": "inputs/nickases/demo.nickases.yaml"},
                    "output": {
                        "run_dir": "outputs/cassettes",
                        "emit_visual_contracts": True,
                        "emit_baserender_jobs": True,
                        "baserender_profiles": ["duplex_qa", "hairpin_qa"],
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return spec_path


def _base_solve_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "topology": {
            "stem5p_arm_pattern": "NNNNNCCTCAGC",
            "loop_pattern": "TTT",
        },
        "construct_context": {
            "left_flank": "",
            "right_flank": "",
            "evaluation_scope": "cassette_plus_flanks",
        },
        "nick_goal": {
            "target_strand": "primary",
            "left_nick_window": {"start": 7, "end": 7},
            "right_nick_window": {"start": 17, "end": 17},
            "bounded_segment_length": {"min": 10, "max": 10},
        },
        "assignment_policy": {
            "allowed_left_variant_ids": ["Nt.BbvCI"],
            "allowed_right_variant_ids": ["Nb.BbvCI"],
            "forbidden_intended_variant_ids": [],
            "forbidden_intended_specificity_ids": [],
            "allow_same_variant": True,
            "allow_same_specificity_opposite_variant": True,
        },
        "site_blacklist": {
            "forbidden_any_site_specificity_ids": [],
            "forbidden_unintended_site_specificity_ids": [],
            "forbidden_any_site_variant_ids": [],
            "scope": "evaluation_context",
        },
        "sequence_blacklist": {
            "forbidden_literals": [],
            "forbidden_iupac_motifs": [],
            "forbid_reverse_complements": True,
            "scope": "evaluation_context",
        },
        "sequence_quality": {},
        "catalog": {"preset": "neb_nicking_v1", "additional_paths": []},
        "search": {
            "max_hits": 3,
            "max_enumerated_candidates": 256,
            "selection": {
                "policy": "greedy_hamming",
                "pool_size": 16,
                "distance_metric": "hamming",
                "min_pairwise_distance": 2,
            },
            "bounded_segment_target": 10,
            "gc_target": 0.5,
            "materialize_top_k": 2,
        },
        "output": {
            "run_dir": "outputs/cassette_solves",
            "emit_visual_contracts": True,
            "emit_baserender_jobs": True,
            "baserender_profiles": ["duplex_qa", "hairpin_qa", "top_hits_duplex_qa", "top_hits_hairpin_qa"],
        },
    }


def _write_solve_spec(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspaces" / "demo_cassette"
    spec_path = workspace / "configs" / "cassettes" / "demo_hairpin.cassette.solve.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump({"cassette_solve": _base_solve_payload()}, sort_keys=False), encoding="utf-8")
    return spec_path


def test_design_publishes_view_bundle_and_baserender_jobs(tmp_path: Path) -> None:
    spec_path = _write_cassette_spec(tmp_path)

    run_dir, report = run_cassette_design(spec_path)

    assert report.status == "satisfied"
    assert (run_dir / "views" / "linear_duplex.v1.json").exists()
    assert (run_dir / "views" / "ssdna_hairpin.v1.json").exists()
    assert (run_dir / "views" / "views_manifest.v1.json").exists()
    assert (run_dir / "baserender_jobs" / "linear_duplex.job.yaml").exists()
    assert (run_dir / "baserender_jobs" / "ssdna_hairpin.job.yaml").exists()


def test_solve_publishes_contracts_jobs_and_renderable_top_hit_sheet(tmp_path: Path) -> None:
    spec_path = _write_solve_spec(tmp_path)

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    assert report.status == "solved"
    assert (run_dir / "views" / "top_hits.linear_duplex.v1.jsonl").exists()
    assert (run_dir / "views" / "top_hits.ssdna_hairpin.v1.jsonl").exists()
    assert (run_dir / "baserender_jobs" / "top_hits_duplex.job.yaml").exists()
    assert (run_dir / "baserender_jobs" / "top_hits_hairpin.job.yaml").exists()

    hit_dirs = sorted((run_dir / "hits").iterdir())
    assert len(hit_dirs) == 2
    first_hit_dir = hit_dirs[0]
    assert (first_hit_dir / "explicit" / "report.json").exists()
    assert (first_hit_dir / "views" / "linear_duplex.v1.json").exists()
    assert (first_hit_dir / "views" / "ssdna_hairpin.v1.json").exists()
    assert (first_hit_dir / "views" / "views_manifest.v1.json").exists()
    assert (first_hit_dir / "baserender_jobs" / "linear_duplex.job.yaml").exists()
    assert (first_hit_dir / "baserender_jobs" / "ssdna_hairpin.job.yaml").exists()

    with (run_dir / "table__hits.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["solution_id"]
    assert rows[0]["views_manifest_path"]
    assert rows[0]["linear_duplex_job_path"]
    assert rows[0]["ssdna_hairpin_job_path"]

    report = baserender.run_job(run_dir / "baserender_jobs" / "top_hits_duplex.job.yaml", caller_root=run_dir)
    assert Path(report.outputs["images_path"]).exists()

    hairpin_report = baserender.run_job(
        first_hit_dir / "baserender_jobs" / "ssdna_hairpin.job.yaml",
        caller_root=run_dir,
    )
    assert Path(hairpin_report.outputs["images_path"]).exists()

    manifest_payload = json.loads((first_hit_dir / "views" / "views_manifest.v1.json").read_text(encoding="utf-8"))
    assert manifest_payload["recommended_jobs"][0]["name"] == "linear_duplex"


def test_solve_omits_view_and_job_paths_when_visual_outputs_are_disabled(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["output"] = {
        "run_dir": "outputs/cassette_solves",
        "emit_visual_contracts": False,
        "emit_baserender_jobs": False,
        "baserender_profiles": [],
    }
    workspace = tmp_path / "workspaces" / "demo_cassette"
    spec_path = workspace / "configs" / "cassettes" / "demo_hairpin.cassette.solve.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump({"cassette_solve": payload}, sort_keys=False), encoding="utf-8")

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    assert report.status == "solved"
    assert not (run_dir / "views" / "top_hits.linear_duplex.v1.jsonl").exists()
    assert not (run_dir / "views" / "top_hits.ssdna_hairpin.v1.jsonl").exists()
    assert not (run_dir / "baserender_jobs" / "top_hits_duplex.job.yaml").exists()
    assert not (run_dir / "baserender_jobs" / "top_hits_hairpin.job.yaml").exists()

    first_hit = report.hits[0]
    assert first_hit.views_manifest_path is None
    assert first_hit.linear_duplex_job_path is None
    assert first_hit.ssdna_hairpin_job_path is None

    first_hit_dir = Path(first_hit.materialized_run_dir)
    assert not (first_hit_dir / "views" / "linear_duplex.v1.json").exists()
    assert not (first_hit_dir / "views" / "ssdna_hairpin.v1.json").exists()
    assert not (first_hit_dir / "views" / "views_manifest.v1.json").exists()
    assert not (first_hit_dir / "baserender_jobs" / "linear_duplex.job.yaml").exists()
    assert not (first_hit_dir / "baserender_jobs" / "ssdna_hairpin.job.yaml").exists()

    with (run_dir / "table__hits.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["views_manifest_path"] == ""
    assert rows[0]["linear_duplex_job_path"] == ""
    assert rows[0]["ssdna_hairpin_job_path"] == ""


def test_solve_keeps_explicit_design_id_consistent_across_csv_jsonl_and_hit_views(tmp_path: Path) -> None:
    spec_path = _write_solve_spec(tmp_path)

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    first_hit = report.hits[0]
    assert first_hit.explicit_design_id is not None
    with (run_dir / "table__hits.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    csv_row = next(row for row in rows if row["solution_id"] == first_hit.solution_id)
    top_hit_rows = (run_dir / "views" / "top_hits.linear_duplex.v1.jsonl").read_text(encoding="utf-8").splitlines()
    top_hit_view = json.loads(top_hit_rows[0])
    per_hit_view = json.loads(
        (Path(first_hit.materialized_run_dir) / "views" / "linear_duplex.v1.json").read_text(encoding="utf-8")
    )

    assert csv_row["explicit_design_id"] == first_hit.explicit_design_id
    assert top_hit_view["meta"]["explicit_design_id"] == first_hit.explicit_design_id
    assert per_hit_view["meta"]["explicit_design_id"] == first_hit.explicit_design_id


def test_solve_with_visuals_disabled_skips_view_contract_builds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _base_solve_payload()
    payload["output"] = {
        "run_dir": "outputs/cassette_solves",
        "emit_visual_contracts": False,
        "emit_baserender_jobs": False,
        "baserender_profiles": [],
    }
    workspace = tmp_path / "workspaces" / "demo_cassette"
    spec_path = workspace / "configs" / "cassettes" / "demo_hairpin.cassette.solve.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump({"cassette_solve": payload}, sort_keys=False), encoding="utf-8")

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("view publication should be skipped when visuals are disabled")

    monkeypatch.setattr(
        "dnadesign.cruncher.app.cassette_solve_workflow.build_linear_duplex_view",
        _unexpected,
    )
    monkeypatch.setattr(
        "dnadesign.cruncher.app.cassette_solve_workflow.build_hairpin_topology_view",
        _unexpected,
    )

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    assert report.status == "solved"
    assert not (run_dir / "views" / "top_hits.linear_duplex.v1.jsonl").exists()


def test_solve_manifest_records_resolved_catalog_provenance(tmp_path: Path) -> None:
    spec_path = _write_solve_spec(tmp_path)

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    assert report.status == "solved"
    manifest = json.loads((run_dir / "solve_manifest.json").read_text(encoding="utf-8"))

    assert manifest["resolved_catalog_path"] == str((run_dir / "specs" / "resolved_catalog.yaml").resolve())
    assert manifest["resolved_catalog_sha256"]
    assert manifest["catalog_preset"] == "neb_nicking_v1"
