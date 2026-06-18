"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_snapback_contract_jobs.py

Tests for snapback visual-contract rendering through the public baserender job.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

import dnadesign.baserender as baserender
from dnadesign.cruncher.app.snapback_solve_workflow import run_snapback_solve
from dnadesign.cruncher.app.snapback_workflow import run_snapback_design


def _write_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspaces" / "demo_snapback"
    spec_path = workspace / "configs" / "snapback" / "demo.snapback.yaml"
    catalog_path = workspace / "inputs" / "nickases" / "local.nickases.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nt.Bpu10I",
                            "specificity_id": "Bpu10I",
                            "motif_top_5to3": "CCTNAGC",
                            "top_cut_offset": 2,
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    spec_path.write_text(
        yaml.safe_dump(
            {
                "snapback": {
                    "schema_version": 2,
                    "contract": "single_nick_snapback_v2",
                    "name": "demo_snapback",
                },
                "input": {
                    "canonical_top_strand": {
                        "sequence": "CCTCAGCA",
                        "protected_region": {"start": 0, "end": 8},
                        "pre_nick_duplex_window": {"start": 0, "end": 8},
                    }
                },
                "design": {
                    "nickase": {
                        "variant_id": "Nt.Bpu10I",
                        "catalog": {"additional_paths": ["inputs/nickases/local.nickases.yaml"]},
                    },
                    "orientation_policy": {
                        "normalize_to_top_strand_nick": True,
                        "release_direction": "left_to_right_from_nick",
                    },
                    "single_nick_goal": {"nick_boundary_window": {"min": 2, "max": 2}},
                    "topology": {
                        "retained_homology_window": {"start": 2, "end": 6},
                        "cap_sequence": "T",
                        "foldback_arm": "CTGA",
                        "homology_policy": {"max_mismatches": 0, "min_paired_bp": 4, "max_paired_bp": 4},
                    },
                    "constraints": {
                        "terminal_ligatable_duplex_bp": {"min": 4, "max": 4},
                        "max_uninterrupted_duplex_bp": 4,
                        "max_added_nt": 5,
                        "forbid_additional_target_strand_nicks": False,
                        "forbid_any_additional_nicks": False,
                    },
                    "sequence_quality": {
                        "gc_fraction": {"min": 0.25, "max": 0.75},
                        "max_homopolymer_run": 2,
                    },
                },
                "output": {
                    "run_dir": "outputs/design",
                    "emit_visual_contracts": True,
                    "emit_baserender_jobs": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return spec_path


def test_snapback_design_emits_one_public_triptych_job_that_renders_a_png(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path)
    run_dir, report = run_snapback_design(spec_path)

    assert report.status == "satisfied"
    job_path = run_dir / "baserender_jobs" / "snapback_triptych.job.yaml"
    render_path = run_dir / "plots" / "snapback_triptych.png"
    job_payload = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    assert job_payload["input"]["kind"] == "jsonl"
    assert job_payload["input"]["adapter"]["kind"] == "snapback_visual_v1"
    assert job_payload["render"]["renderer"] == "snapback_map"
    report = baserender.run_job(job_path, caller_root=run_dir)
    assert Path(report.outputs["images_path"]) == render_path.resolve()
    assert render_path.exists()


def test_snapback_solve_materialized_hit_emits_triptych_job_that_renders_a_png(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "demo_snapback"
    solve_path = workspace / "configs" / "snapback" / "demo.snapback.solve.yaml"
    catalog_path = workspace / "inputs" / "nickases" / "local.nickases.yaml"
    solve_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nt.Bpu10I",
                            "specificity_id": "Bpu10I",
                            "motif_top_5to3": "CCTNAGC",
                            "top_cut_offset": 2,
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    solve_path.write_text(
        yaml.safe_dump(
            {
                "snapback_solve": {
                    "schema_version": 3,
                    "contract": "single_nick_snapback_solve_v3",
                    "name": "demo_snapback_solve",
                },
                "input": {
                    "canonical_top_strand": {
                        "sequence": "CCTCAGCA",
                        "protected_region": {"start": 0, "end": 8},
                        "pre_nick_duplex_window": {"start": 0, "end": 8},
                    }
                },
                "catalog": {"additional_paths": ["inputs/nickases/local.nickases.yaml"]},
                "orientation_policy": {"normalize_to_top_strand_nick": True},
                "goal": {"nick_boundary_window": {"min": 2, "max": 2}},
                "search": {
                    "retained_homology_length": {"min": 4, "max": 4},
                    "max_added_nt": 5,
                    "max_mismatches": 0,
                    "max_enumerated_candidates": 64,
                    "max_search_nodes": 64,
                    "max_hits": 2,
                    "materialize_top_k": 1,
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
                "output": {
                    "run_dir": "outputs/solve",
                    "emit_visual_contracts": True,
                    "emit_baserender_jobs": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    run_dir, report = run_snapback_solve(solve_path)

    assert report.status == "satisfied"
    hit_dir = run_dir / "analysis" / "materialized_hits" / "hit_01"
    job_path = hit_dir / "baserender_jobs" / "snapback_triptych.job.yaml"
    render_path = hit_dir / "plots" / "snapback_triptych.png"
    job_payload = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    assert job_payload["input"]["kind"] == "jsonl"
    assert job_payload["input"]["adapter"]["kind"] == "snapback_visual_v1"
    assert job_payload["render"]["renderer"] == "snapback_map"
    render_report = baserender.run_job(job_path, caller_root=hit_dir)
    assert Path(render_report.outputs["images_path"]) == render_path.resolve()
    assert render_path.exists()
