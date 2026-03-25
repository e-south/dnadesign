"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cassette/test_solver.py

Solve/search tests for the cassette workflow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from dnadesign.cruncher.app.cassette_solve_workflow import run_cassette_solve, solve_cassette_spec
from dnadesign.cruncher.cassette.errors import CassetteSpecError
from dnadesign.cruncher.cassette.load import load_cassette_solve_spec


def _write_solve_spec(tmp_path: Path, *, payload: dict[str, Any]) -> Path:
    workspace = tmp_path / "workspaces" / "demo_cassette"
    spec_path = workspace / "configs" / "cassettes" / "demo_hairpin.cassette.solve.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump({"cassette_solve": payload}, sort_keys=False), encoding="utf-8")
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
            "left_nick_window": {"start": 0, "end": 0},
            "right_nick_window": {"start": 24, "end": 24},
            "bounded_segment_length": {"min": 24, "max": 24},
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
        "catalog": {
            "preset": "neb_nicking_v1",
            "additional_paths": [],
        },
        "search": {
            "max_hits": 3,
            "max_enumerated_candidates": 256,
            "min_pairwise_hamming_distance": 2,
            "bounded_segment_target": 24,
            "gc_target": 0.5,
            "materialize_top_k": 2,
        },
        "output": {
            "run_dir": "outputs/cassette_solves",
            "write_render_contract": True,
        },
    }


def test_load_cassette_solve_spec_requires_target_strand(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["nick_goal"].pop("target_strand")
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    with pytest.raises(CassetteSpecError, match="target_strand"):
        load_cassette_solve_spec(spec_path)


def test_load_cassette_solve_spec_rejects_duplicate_assignment_ids(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["assignment_policy"]["allowed_left_variant_ids"] = ["Nt.BbvCI", "Nt.BbvCI"]
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    with pytest.raises(CassetteSpecError, match="must not repeat values"):
        load_cassette_solve_spec(spec_path)


def test_load_cassette_solve_spec_rejects_oversized_assignment_pair_space(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["assignment_policy"]["allowed_left_variant_ids"] = [f"left_{index}" for index in range(1, 18)]
    payload["assignment_policy"]["allowed_right_variant_ids"] = [f"right_{index}" for index in range(1, 18)]
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    with pytest.raises(CassetteSpecError, match="assignment pair space"):
        load_cassette_solve_spec(spec_path)


def test_load_cassette_solve_spec_rejects_oversized_topology(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["topology"]["stem5p_arm_pattern"] = "N" * 65
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    with pytest.raises(CassetteSpecError, match="stem5p_arm_pattern"):
        load_cassette_solve_spec(spec_path)


def test_load_cassette_solve_spec_rejects_unsafe_search_caps(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["search"]["max_hits"] = 129
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    with pytest.raises(CassetteSpecError, match="max_hits"):
        load_cassette_solve_spec(spec_path)


def test_solve_returns_multiple_hits_and_materializes_top_k(tmp_path: Path) -> None:
    spec_path = _write_solve_spec(tmp_path, payload=_base_solve_payload())

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    assert report.status == "solved"
    assert report.solve_id is not None
    assert len(report.hits) == 3
    assert report.metadata.materialized_hit_count == 2
    assert report.hits[0].materialized_run_dir is not None
    assert (run_dir / "solve_report.json").exists()
    assert (run_dir / "solve_manifest.json").exists()
    assert (run_dir / "solve_status.json").exists()
    assert (run_dir / "table__hits.csv").exists()
    assert (run_dir / "specs" / "resolved_catalog.yaml").exists()
    hit_dirs = sorted((run_dir / "hits").iterdir())
    assert len(hit_dirs) == 2
    first_hit_report = json.loads((hit_dirs[0] / "report.json").read_text(encoding="utf-8"))
    assert first_hit_report["status"] == "satisfied"
    assert (hit_dirs[0] / "resolved_candidate.cassette.yaml").exists()
    assert (hit_dirs[0] / "render_contract.json").exists()


def test_solve_reports_no_hits_when_blacklist_blocks_required_site(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["sequence_blacklist"]["forbidden_literals"] = ["CCTCAGC"]
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    assert report.status == "no_hits"
    assert report.hits == []
    assert (run_dir / "solve_report.json").exists()
    hit_table = (run_dir / "table__hits.csv").read_text(encoding="utf-8")
    assert "rank,score,hit_id" in hit_table


def test_solve_reports_invalid_spec_when_assignment_policy_eliminates_all_pairs(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["assignment_policy"]["forbidden_intended_variant_ids"] = ["Nt.BbvCI", "Nb.BbvCI"]
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    report = solve_cassette_spec(spec_path)

    assert report.status == "invalid_spec"
    assert report.issues[0].code == "NO_ALLOWED_VARIANT_PAIRS"


def test_solve_site_blacklist_operates_at_specificity_level(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["site_blacklist"]["forbidden_any_site_specificity_ids"] = ["BbvCI"]
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    report = solve_cassette_spec(spec_path)

    assert report.status == "no_hits"


def test_solve_search_node_guardrail_surfaces_warning(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["search"]["max_search_nodes"] = 1
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    report = solve_cassette_spec(spec_path)

    assert report.status == "no_hits"
    assert "search.max_search_nodes reached before exhausting the solve search tree." in report.metadata.warnings


def test_run_cassette_solve_persists_invalid_spec_bundle(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["nick_goal"].pop("target_strand")
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    assert report.status == "invalid_spec"
    assert report.run_dir == str(run_dir.resolve())
    assert (run_dir / "solve_report.json").exists()
    assert (run_dir / "solve_manifest.json").exists()
    assert (run_dir / "solve_status.json").exists()
    assert (run_dir / "table__hits.csv").exists()
    assert (run_dir / "specs" / "input_solve_spec.yaml").exists()
    assert not (run_dir / "specs" / "resolved_catalog.yaml").exists()


def test_run_cassette_solve_persists_invalid_catalog_bundle(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["catalog"]["additional_paths"] = ["inputs/catalogs/missing.yaml"]
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    assert report.status == "invalid_catalog"
    assert report.run_dir == str(run_dir.resolve())
    assert (run_dir / "solve_report.json").exists()
    assert (run_dir / "solve_manifest.json").exists()
    assert (run_dir / "solve_status.json").exists()
    assert (run_dir / "table__hits.csv").exists()
    assert (run_dir / "specs" / "input_solve_spec.yaml").exists()
    assert not (run_dir / "specs" / "resolved_catalog.yaml").exists()


def test_run_cassette_solve_status_surface_marks_truncated_search(tmp_path: Path) -> None:
    payload = _base_solve_payload()
    payload["search"]["max_enumerated_candidates"] = 1
    payload["search"]["materialize_top_k"] = 1
    spec_path = _write_solve_spec(tmp_path, payload=payload)

    run_dir, report = run_cassette_solve(spec_path)

    assert run_dir is not None
    assert report.status == "solved"
    status_payload = json.loads((run_dir / "solve_status.json").read_text(encoding="utf-8"))
    assert status_payload["search_truncated"] is True
    assert (
        "search.max_enumerated_candidates reached before exhausting the solve search space."
        in status_payload["warnings"]
    )


def test_solve_tracks_total_accepted_candidates_even_when_hit_buffer_truncates(tmp_path: Path) -> None:
    spec_path = _write_solve_spec(tmp_path, payload=_base_solve_payload())

    report = solve_cassette_spec(spec_path)

    assert report.status == "solved"
    assert "internal hit buffer truncated to keep solve memory bounded." in report.metadata.warnings
    assert report.metadata.accepted_candidate_count > 128
