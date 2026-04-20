"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_snapback_cli.py

CLI contract tests for the snapback command group.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def _write_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspaces" / "demo_snapback"
    explicit_path = workspace / "configs" / "snapback" / "demo.snapback.yaml"
    solve_path = workspace / "configs" / "snapback" / "demo.snapback.solve.yaml"
    catalog_path = workspace / "inputs" / "nickases" / "local.nickases.yaml"
    explicit_path.parent.mkdir(parents=True, exist_ok=True)
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
    explicit_path.write_text(
        yaml.safe_dump(
            {
                "snapback": {
                    "schema_version": 2,
                    "contract": "single_nick_snapback_v2",
                    "name": "demo_snapback",
                },
                "input": {
                    "canonical_top_strand": {
                        "sequence": "CCTCAGCAGTC",
                        "protected_region": {"start": 0, "end": 11},
                        "pre_nick_duplex_window": {"start": 0, "end": 11},
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
                        "retained_homology_window": {"start": 7, "end": 11},
                        "cap_sequence": "TT",
                        "foldback_arm": "GACT",
                        "homology_policy": {"max_mismatches": 0, "min_paired_bp": 4, "max_paired_bp": 4},
                    },
                    "constraints": {
                        "terminal_ligatable_duplex_bp": {"min": 4, "max": 4},
                        "max_uninterrupted_duplex_bp": 4,
                        "max_added_nt": 6,
                        "forbid_additional_target_strand_nicks": False,
                        "forbid_any_additional_nicks": False,
                    },
                    "sequence_quality": {
                        "gc_fraction": {"min": 0.25, "max": 0.75},
                        "max_homopolymer_run": 2,
                    },
                },
                "output": {
                    "run_dir": "outputs/snapback",
                    "emit_visual_contracts": True,
                    "emit_baserender_jobs": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    solve_path.write_text(
        yaml.safe_dump(
            {
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
                "output": {
                    "run_dir": "outputs/snapback_solves",
                    "emit_visual_contracts": True,
                    "emit_baserender_jobs": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return workspace, explicit_path, solve_path


def test_root_help_includes_snapback_group() -> None:
    result = runner.invoke(app, ["--help"], color=False)

    assert result.exit_code == 0
    assert "snapback" in result.output
    assert "single-nick snapback" in result.output


def test_snapback_help_describes_workspace_validate_design_solve_and_show_surface() -> None:
    result = runner.invoke(app, ["snapback", "--help"], color=False)

    assert result.exit_code == 0
    assert "init-workspace" in result.output
    assert "validate" in result.output
    assert "design" in result.output
    assert "solve" in result.output
    assert "show" in result.output


def test_snapback_command_module_defers_workflow_import() -> None:
    command_module = "dnadesign.cruncher.cli.commands.snapback"
    explicit_workflow_module = "dnadesign.cruncher.app.snapback_workflow"
    solve_workflow_module = "dnadesign.cruncher.app.snapback_solve_workflow"
    sys.modules.pop(command_module, None)
    sys.modules.pop(explicit_workflow_module, None)
    sys.modules.pop(solve_workflow_module, None)

    importlib.import_module(command_module)

    assert explicit_workflow_module not in sys.modules
    assert solve_workflow_module not in sys.modules


def test_snapback_validate_json_reports_v2_candidate_metrics(tmp_path: Path) -> None:
    _workspace, explicit_path, _solve_path = _write_workspace(tmp_path)

    result = runner.invoke(app, ["snapback", "validate", "--spec", str(explicit_path), "--json"], color=False)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "satisfied"
    assert payload["candidate"]["nick_boundary"] == 2
    assert payload["candidate"]["released_prefix_nt"] == 5
    assert payload["candidate"]["cap_nt"] == 2


def test_snapback_validate_returns_typed_invalid_catalog_report(tmp_path: Path) -> None:
    _workspace, explicit_path, _solve_path = _write_workspace(tmp_path)
    payload = yaml.safe_load(explicit_path.read_text(encoding="utf-8"))
    payload["design"]["nickase"]["variant_id"] = "Nt.DoesNotExist"
    explicit_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["snapback", "validate", "--spec", str(explicit_path), "--json"], color=False)

    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["status"] == "invalid_catalog"
    assert report["issues"][0]["code"] == "UNKNOWN_VARIANT_ID"


def test_snapback_validate_returns_typed_invalid_catalog_report_for_missing_catalog_path(tmp_path: Path) -> None:
    _workspace, explicit_path, _solve_path = _write_workspace(tmp_path)
    payload = yaml.safe_load(explicit_path.read_text(encoding="utf-8"))
    payload["design"]["nickase"]["catalog"]["additional_paths"] = ["inputs/nickases/missing.nickases.yaml"]
    explicit_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["snapback", "validate", "--spec", str(explicit_path), "--json"], color=False)

    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["status"] == "invalid_catalog"
    assert report["issues"][0]["code"] == "CATALOG_LOAD_FAILED"


def test_snapback_design_returns_typed_invalid_catalog_report_before_materialization(tmp_path: Path) -> None:
    workspace, explicit_path, _solve_path = _write_workspace(tmp_path)
    payload = yaml.safe_load(explicit_path.read_text(encoding="utf-8"))
    payload["design"]["nickase"]["catalog"]["additional_paths"] = ["inputs/nickases/missing.nickases.yaml"]
    explicit_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["snapback", "design", "--spec", str(explicit_path), "--json"], color=False)

    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["status"] == "invalid_catalog"
    assert report["issues"][0]["code"] == "CATALOG_LOAD_FAILED"
    assert not (workspace / "outputs" / "snapback" / "demo_snapback").exists()


def test_snapback_design_writes_v2_artifacts_and_show_reads_them(tmp_path: Path) -> None:
    workspace, explicit_path, _solve_path = _write_workspace(tmp_path)

    result = runner.invoke(app, ["snapback", "design", "--spec", str(explicit_path)], color=False)

    assert result.exit_code == 0
    run_root = workspace / "outputs" / "snapback" / "demo_snapback"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "meta" / "snapback_manifest.json").exists()
    assert (run_dir / "meta" / "snapback_status.json").exists()
    assert (run_dir / "analysis" / "reports" / "report.json").exists()
    assert (run_dir / "analysis" / "reports" / "report.md").exists()
    assert (run_dir / "export" / "table__candidates.csv").exists()
    assert (run_dir / "provenance" / "nickase_catalog.yaml").exists()
    assert (run_dir / "views" / "pre_nick_duplex.v1.json").exists()
    assert (run_dir / "views" / "post_nick_exposed.v1.json").exists()
    assert (run_dir / "views" / "post_nick_foldback.v1.json").exists()
    assert (run_dir / "views" / "pre_nick_duplex.snapback_visual.v1.json").exists()
    assert (run_dir / "views" / "post_nick_exposed.snapback_visual.v1.json").exists()
    assert (run_dir / "views" / "post_nick_foldback.snapback_visual.v1.json").exists()
    assert (run_dir / "baserender_jobs" / "pre_nick_duplex.job.yaml").exists()
    assert (run_dir / "baserender_jobs" / "post_nick_exposed.job.yaml").exists()
    assert (run_dir / "baserender_jobs" / "post_nick_foldback.job.yaml").exists()

    show_result = runner.invoke(app, ["snapback", "show", "--run", str(run_dir), "--json"], color=False)
    assert show_result.exit_code == 0
    payload = json.loads(show_result.output)
    assert payload["kind"] == "explicit"
    assert payload["spec_name"] == "demo_snapback"
    assert payload["status"] == "satisfied"
    assert payload["pre_nick_duplex_view"] is not None
    assert payload["post_nick_exposed_view"] is not None
    assert payload["post_nick_foldback_view"] is not None
    assert payload["pre_nick_duplex_visual_contract"] is not None
    assert payload["post_nick_exposed_visual_contract"] is not None
    assert payload["post_nick_foldback_visual_contract"] is not None
    assert payload["pre_nick_duplex_job"] is not None
    assert payload["post_nick_exposed_job"] is not None
    assert payload["post_nick_foldback_job"] is not None

    pre_nick_view = json.loads((run_dir / "views" / "pre_nick_duplex.v1.json").read_text(encoding="utf-8"))
    assert pre_nick_view["kind"] == "snapback_pre_nick_duplex_v1"
    assert pre_nick_view["rows"]["top"]["label"] == "Canonical top strand 5' -> 3'"
    assert pre_nick_view["rows"]["complement"]["label"] == "Complement strand 3' -> 5'"
    assert pre_nick_view["nick_boundary"] == 2
    assert pre_nick_view["ligation_junction_boundary"] == 7

    exposed_view = json.loads((run_dir / "views" / "post_nick_exposed.v1.json").read_text(encoding="utf-8"))
    assert exposed_view["kind"] == "snapback_post_nick_exposed_v1"
    assert exposed_view["topology"]["released_prefix_span"]["start"] == 2
    assert exposed_view["ligation_junction_boundary"] == 7

    foldback_view = json.loads((run_dir / "views" / "post_nick_foldback.v1.json").read_text(encoding="utf-8"))
    assert foldback_view["kind"] == "snapback_post_nick_foldback_v1"
    assert foldback_view["source_nick_boundary"] == 2
    assert foldback_view["ligation_junction_boundary"] == 5
    assert foldback_view["primary_mismatch_positions"] == []

    visual_contract = json.loads(
        (run_dir / "views" / "post_nick_foldback.snapback_visual.v1.json").read_text(encoding="utf-8")
    )
    assert visual_contract["contract_kind"] == "snapback_visual_v1"
    assert visual_contract["state_kind"] == "post_nick_foldback"
    assert visual_contract["ligation_junction_boundary"] == 5
    assert any(pairing["left_index"] == 5 for pairing in visual_contract["pairings"])

    pre_visual_contract = json.loads(
        (run_dir / "views" / "pre_nick_duplex.snapback_visual.v1.json").read_text(encoding="utf-8")
    )
    assert pre_visual_contract["pairings"] == []

    exposed_visual_contract = json.loads(
        (run_dir / "views" / "post_nick_exposed.snapback_visual.v1.json").read_text(encoding="utf-8")
    )
    assert exposed_visual_contract["pairings"] == []
    assert exposed_visual_contract["exposed_complement_span"] == {
        "start": 2,
        "end": len(exposed_visual_contract["primary_sequence"]),
    }

    views_manifest = json.loads((run_dir / "views" / "views_manifest.v1.json").read_text(encoding="utf-8"))
    assert len(views_manifest["recommended_jobs"]) == 3


def test_snapback_show_fails_fast_when_manifest_and_status_disagree(tmp_path: Path) -> None:
    workspace, explicit_path, _solve_path = _write_workspace(tmp_path)

    design_result = runner.invoke(app, ["snapback", "design", "--spec", str(explicit_path)], color=False)
    assert design_result.exit_code == 0
    run_root = workspace / "outputs" / "snapback" / "demo_snapback"
    run_dir = next(iter(run_root.iterdir()))
    status_path = run_dir / "meta" / "snapback_status.json"
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    status_payload["status"] = "unsatisfied"
    status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

    show_result = runner.invoke(app, ["snapback", "show", "--run", str(run_dir)], color=False)

    assert show_result.exit_code == 1
    assert "manifest/status status drift" in show_result.output


def test_snapback_solve_writes_artifacts_and_show_reads_solve_bundle(tmp_path: Path) -> None:
    workspace, _explicit_path, solve_path = _write_workspace(tmp_path)

    result = runner.invoke(app, ["snapback", "solve", "--spec", str(solve_path)], color=False)

    assert result.exit_code == 0
    run_root = workspace / "outputs" / "snapback_solves"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "solve_report.json").exists()
    assert (run_dir / "solve_report.md").exists()
    assert (run_dir / "table__hits.csv").exists()
    assert (run_dir / "solve_manifest.json").exists()
    assert (run_dir / "solve_status.json").exists()
    assert (run_dir / "specs" / "resolved_catalog.yaml").exists()
    assert (run_dir / "hits").exists()

    show_result = runner.invoke(app, ["snapback", "show", "--run", str(run_dir), "--json"], color=False)
    assert show_result.exit_code == 0
    payload = json.loads(show_result.output)
    assert payload["kind"] == "solve"
    assert payload["status"] == "satisfied"
    assert payload["solve_report"] is not None
    materialized_dirs = sorted((run_dir / "hits").iterdir())
    assert len(materialized_dirs) == 2
    assert (materialized_dirs[0] / "views" / "post_nick_foldback.snapback_visual.v1.json").exists()
    assert (materialized_dirs[0] / "baserender_jobs" / "post_nick_foldback.job.yaml").exists()


def test_snapback_init_workspace_scaffolds_v2_examples(tmp_path: Path) -> None:
    target = tmp_path / "workspaces" / "demo_snapback"

    result = runner.invoke(app, ["snapback", "init-workspace", "--output", str(target)], color=False)

    assert result.exit_code == 0
    assert (target / "README.md").exists()
    assert (target / "configs" / "snapback" / "demo_teto_bpu10i_cap.snapback.yaml").exists()
    assert (target / "configs" / "snapback" / "demo_teto_bpu10i_cap.snapback.solve.yaml").exists()
    assert (target / "inputs" / "nickases" / "local.nickases.yaml").exists()
