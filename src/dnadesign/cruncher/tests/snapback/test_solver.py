"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/snapback/test_solver.py

Bounded solve tests for v2 snapback workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from dnadesign.cruncher.app.snapback_solve_workflow import run_snapback_solve


def _write_workspace(
    tmp_path: Path,
    *,
    spec_payload: dict[str, object],
    catalog_entries: list[dict[str, object]],
) -> Path:
    workspace = tmp_path / "workspaces" / "demo_snapback"
    spec_path = workspace / "configs" / "snapback" / "demo.snapback.solve.yaml"
    catalog_path = workspace / "inputs" / "nickases" / "local.nickases.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        yaml.safe_dump({"nickases": {"schema_version": 1, "entries": catalog_entries}}, sort_keys=False),
        encoding="utf-8",
    )
    spec_path.write_text(yaml.safe_dump(spec_payload, sort_keys=False), encoding="utf-8")
    return spec_path


def _catalog_entries() -> list[dict[str, object]]:
    return [
        {
            "id": "Nt.Bpu10I",
            "specificity_id": "Bpu10I",
            "motif_top_5to3": "CCTNAGC",
            "top_cut_offset": 2,
            "source": "demo",
        }
    ]


def _base_payload() -> dict[str, object]:
    return {
        "snapback_solve": {
            "schema_version": 2,
            "contract": "single_nick_snapback_solve_v2",
            "name": "demo_snapback_solve",
        },
        "input": {
            "canonical_top_strand": {
                "sequence": "CCTCAGCAGTC",
                "protected_region": {"start": 0, "end": 11},
                "pre_nick_duplex_window": {"start": 0, "end": 11},
            }
        },
        "catalog": {"additional_paths": ["inputs/nickases/local.nickases.yaml"]},
        "nickase_policy": {
            "allowed_variant_ids": ["Nt.Bpu10I"],
            "normalize_to_top_strand_nick": True,
        },
        "goal": {
            "nick_boundary_window": {"min": 2, "max": 2},
            "retained_start_from_nick": {"min": 5, "max": 5},
        },
        "search": {
            "retained_homology_length": {"min": 4, "max": 4},
            "cap_nt": {"min": 1, "max": 1},
            "max_added_nt": 5,
            "max_mismatches": 0,
            "max_enumerated_candidates": 64,
            "max_search_nodes": 64,
            "max_hits": 4,
            "materialize_top_k": 2,
        },
        "constraints": {
            "terminal_ligatable_duplex_bp": {"min": 4, "max": 4},
            "max_uninterrupted_duplex_bp": 4,
            "forbid_additional_target_strand_nicks": False,
            "forbid_any_additional_nicks": False,
        },
        "sequence_quality": {
            "gc_fraction": {"min": 0.0, "max": 0.75},
            "max_homopolymer_run": 3,
        },
        "output": {"run_dir": "outputs/snapback_solves", "emit_visual_contracts": True},
    }


def test_run_snapback_solve_finds_ranked_hits_and_materializes_top_k(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path, spec_payload=_base_payload(), catalog_entries=_catalog_entries())

    run_dir, report = run_snapback_solve(spec_path)

    assert report.status == "satisfied"
    assert len(report.hits) == 4
    assert report.hits[0].cap_sequence == "A"
    assert report.hits[0].nick_boundary_from_left == 2
    assert report.hits[0].retained_start_from_nick == 5
    assert report.metadata.materialized_hit_count == 2
    assert (run_dir / "solve_report.json").exists()
    assert (run_dir / "solve_report.md").exists()
    assert (run_dir / "table__hits.csv").exists()
    assert (run_dir / "solve_manifest.json").exists()
    assert (run_dir / "solve_status.json").exists()
    assert (run_dir / "specs" / "input_solve_spec.yaml").exists()
    assert (run_dir / "specs" / "resolved_catalog.yaml").exists()
    materialized_dirs = sorted((run_dir / "hits").iterdir())
    assert len(materialized_dirs) == 2
    first_payload = json.loads(
        (materialized_dirs[0] / "analysis" / "reports" / "report.json").read_text(encoding="utf-8")
    )
    assert first_payload["candidate"]["cap_sequence"] == "A"
    report_md = (materialized_dirs[0] / "analysis" / "reports" / "report.md").read_text(encoding="utf-8")
    assert "nick_boundary_from_left: 2" in report_md
    assert "extra_nick_event_count: 0" in report_md


def test_run_snapback_solve_marks_truncation_when_bounds_are_hit(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["search"]["max_enumerated_candidates"] = 1
    payload["search"]["max_search_nodes"] = 1
    spec_path = _write_workspace(tmp_path, spec_payload=payload, catalog_entries=_catalog_entries())

    run_dir, report = run_snapback_solve(spec_path)

    assert report.status == "search_truncated"
    assert report.metadata.search_truncated is True
    assert report.metadata.enumerated_candidate_count == 1
    status_payload = json.loads((run_dir / "solve_status.json").read_text(encoding="utf-8"))
    assert status_payload["status"] == "search_truncated"


def test_run_snapback_solve_does_not_bias_gc_when_gc_bounds_are_omitted(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["sequence_quality"]["gc_fraction"] = None
    spec_path = _write_workspace(tmp_path, spec_payload=payload, catalog_entries=_catalog_entries())

    _run_dir, report = run_snapback_solve(spec_path)

    assert report.status == "satisfied"
    assert [hit.cap_sequence for hit in report.hits] == ["A", "C", "T", "G"]
