"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_snapback_cli.py

CLI contract tests for the snapback command group.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.nickases.models import reverse_complement

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
                    "run_dir": "outputs/solve",
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
    assert "visual" in result.output
    assert "solve" in result.output
    assert "target-search" in result.output
    assert "show" in result.output


def test_snapback_solve_help_describes_v3_codesign_contract() -> None:
    result = runner.invoke(app, ["snapback", "solve", "--help"], color=False)

    assert result.exit_code == 0
    assert "v3 co-design solve spec" in result.output


def test_snapback_visual_help_describes_visual_spec_contract() -> None:
    result = runner.invoke(app, ["snapback", "visual", "--help"], color=False)

    assert result.exit_code == 0
    assert "visual-only snapback example" in result.output


def test_snapback_target_search_json_reports_exact_and_near_hits(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces" / "demo_snapback"
    workspace_root.mkdir(parents=True, exist_ok=True)

    result = runner.invoke(
        app,
        ["snapback", "target-search", "--workspace-root", str(workspace_root), "--json"],
        color=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "exact_hits_found"
    assert payload["exact_hits"][0]["variant_id"] == "Nb.BsrDI"
    assert payload["exact_hits"][0]["nick_boundary_from_left"] == 0
    assert {hit["variant_id"] for hit in payload["exact_hits"]} == {"Nb.BsrDI", "Nb.BtsI", "Nt.CviPII"}
    assert any(hit["variant_id"] == "Nt.Bpu10I" and hit["nick_boundary_from_left"] == 2 for hit in payload["near_hits"])


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
    assert payload["candidate"]["released_prefix_nt"] == 0
    assert payload["candidate"]["cap_nt"] == 3
    assert payload["candidate"]["cap_extension_nt"] == 1


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
    assert not (workspace / "outputs" / "design").exists()


def test_snapback_design_writes_v2_artifacts_and_show_reads_them(tmp_path: Path) -> None:
    workspace, explicit_path, _solve_path = _write_workspace(tmp_path)

    result = runner.invoke(app, ["snapback", "design", "--spec", str(explicit_path)], color=False)

    assert result.exit_code == 0
    run_dir = workspace / "outputs" / "design"
    assert (run_dir / "meta" / "snapback_manifest.json").exists()
    assert (run_dir / "meta" / "snapback_status.json").exists()
    assert (run_dir / "analysis" / "reports" / "report.json").exists()
    assert (run_dir / "analysis" / "reports" / "report.md").exists()
    assert (run_dir / "export" / "table__candidates.csv").exists()
    assert (run_dir / "provenance" / "nickase_catalog.yaml").exists()
    assert (run_dir / "analysis" / "views" / "pre_nick_duplex.v1.json").exists()
    assert (run_dir / "analysis" / "views" / "post_nick_exposed.v1.json").exists()
    assert (run_dir / "analysis" / "views" / "post_nick_foldback.v1.json").exists()
    assert (run_dir / "analysis" / "views" / "pre_nick_duplex.snapback_visual.v1.json").exists()
    assert (run_dir / "analysis" / "views" / "post_nick_exposed.snapback_visual.v1.json").exists()
    assert (run_dir / "analysis" / "views" / "post_nick_foldback.snapback_visual.v1.json").exists()
    assert (run_dir / "analysis" / "views" / "snapback_triptych.snapback_visual.v1.jsonl").exists()
    assert (run_dir / "baserender_jobs" / "snapback_triptych.job.yaml").exists()
    assert not (run_dir / "plots").exists()

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
    assert payload["snapback_triptych_visual_contracts"] is not None
    assert payload["snapback_triptych_job"] is not None
    assert payload["plots_dir"] is None

    pre_nick_view = json.loads((run_dir / "analysis" / "views" / "pre_nick_duplex.v1.json").read_text(encoding="utf-8"))
    assert pre_nick_view["kind"] == "snapback_pre_nick_duplex_v1"
    assert pre_nick_view["rows"]["top"]["label"] == "Canonical top strand 5' -> 3'"
    assert pre_nick_view["rows"]["complement"]["label"] == "Complement strand 3' -> 5'"
    assert pre_nick_view["nick_boundary"] == 2
    assert pre_nick_view["ligation_junction_boundary"] == 2

    exposed_view = json.loads(
        (run_dir / "analysis" / "views" / "post_nick_exposed.v1.json").read_text(encoding="utf-8")
    )
    assert exposed_view["kind"] == "snapback_post_nick_exposed_v1"
    assert exposed_view["topology"]["released_prefix_span"]["start"] == 2
    assert exposed_view["topology"]["released_prefix_span"]["end"] == 2
    assert exposed_view["ligation_junction_boundary"] == 2

    foldback_view = json.loads(
        (run_dir / "analysis" / "views" / "post_nick_foldback.v1.json").read_text(encoding="utf-8")
    )
    assert foldback_view["kind"] == "snapback_post_nick_foldback_v1"
    assert foldback_view["source_nick_boundary"] == 2
    assert foldback_view["ligation_junction_boundary"] == 0
    assert foldback_view["primary_mismatch_positions"] == []

    visual_contract = json.loads(
        (run_dir / "analysis" / "views" / "post_nick_foldback.snapback_visual.v1.json").read_text(encoding="utf-8")
    )
    assert visual_contract["contract_kind"] == "snapback_visual_v1"
    assert visual_contract["state_kind"] == "post_nick_foldback"
    assert visual_contract["ligation_junction_boundary"] == 0
    assert visual_contract["loop_geometry"]["kind"] == "hairpin_corner_triloop_v1"
    assert visual_contract["loop_geometry"]["source_cap_span"] == {"start": 4, "end": 6}
    assert visual_contract["loop_geometry"]["cap_extension_span"] == {"start": 6, "end": 7}
    assert visual_contract["complement_sequence"] == reverse_complement(visual_contract["primary_sequence"])[::-1]
    assert visual_contract["complement_sequence"] != visual_contract["primary_sequence"]
    assert any(pairing["left_index"] == 0 for pairing in visual_contract["pairings"])

    pre_visual_contract = json.loads(
        (run_dir / "analysis" / "views" / "pre_nick_duplex.snapback_visual.v1.json").read_text(encoding="utf-8")
    )
    assert pre_visual_contract["pairings"] == []

    exposed_visual_contract = json.loads(
        (run_dir / "analysis" / "views" / "post_nick_exposed.snapback_visual.v1.json").read_text(encoding="utf-8")
    )
    assert exposed_visual_contract["pairings"] == []
    assert exposed_visual_contract["exposed_complement_span"] == {
        "start": 2,
        "end": len(exposed_visual_contract["primary_sequence"]),
    }

    views_manifest = json.loads((run_dir / "analysis" / "views" / "views_manifest.v1.json").read_text(encoding="utf-8"))
    assert len(views_manifest["recommended_jobs"]) == 1
    assert views_manifest["recommended_jobs"][0]["name"] == "snapback_triptych"


def test_snapback_show_fails_fast_when_manifest_and_status_disagree(tmp_path: Path) -> None:
    workspace, explicit_path, _solve_path = _write_workspace(tmp_path)

    design_result = runner.invoke(app, ["snapback", "design", "--spec", str(explicit_path)], color=False)
    assert design_result.exit_code == 0
    run_dir = workspace / "outputs" / "design"
    status_path = run_dir / "meta" / "snapback_status.json"
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    status_payload["status"] = "unsatisfied"
    status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

    show_result = runner.invoke(app, ["snapback", "show", "--run", str(run_dir)], color=False)

    assert show_result.exit_code == 1
    assert "manifest/status status drift" in show_result.output


def test_snapback_show_fails_when_explicit_report_run_dir_drifts(tmp_path: Path) -> None:
    workspace, explicit_path, _solve_path = _write_workspace(tmp_path)

    design_result = runner.invoke(app, ["snapback", "design", "--spec", str(explicit_path)], color=False)
    assert design_result.exit_code == 0
    run_dir = workspace / "outputs" / "design"
    report_path = run_dir / "analysis" / "reports" / "report.json"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["run_dir"] = "/tmp/drifted"
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    show_result = runner.invoke(app, ["snapback", "show", "--run", str(run_dir)], color=False)

    assert show_result.exit_code == 1
    assert "Snapback report run_dir drift detected." in show_result.output


def test_snapback_solve_writes_artifacts_and_show_reads_solve_bundle(tmp_path: Path) -> None:
    workspace, _explicit_path, solve_path = _write_workspace(tmp_path)

    result = runner.invoke(app, ["snapback", "solve", "--spec", str(solve_path)], color=False)

    assert result.exit_code == 0
    run_dir = workspace / "outputs" / "solve"
    assert (run_dir / "analysis" / "reports" / "solve_report.json").exists()
    assert (run_dir / "analysis" / "reports" / "solve_report.md").exists()
    assert (run_dir / "export" / "table__hits.csv").exists()
    assert (run_dir / "export" / "table__frontier.csv").exists()
    assert (run_dir / "meta" / "solve_manifest.json").exists()
    assert (run_dir / "meta" / "solve_status.json").exists()
    assert (run_dir / "provenance" / "resolved_catalog.yaml").exists()
    assert (run_dir / "analysis" / "materialized_hits").exists()
    assert not (run_dir / "analysis" / "views").exists()
    assert not (run_dir / "baserender_jobs").exists()
    assert not (run_dir / "plots").exists()

    show_result = runner.invoke(app, ["snapback", "show", "--run", str(run_dir), "--json"], color=False)
    assert show_result.exit_code == 0
    payload = json.loads(show_result.output)
    assert payload["kind"] == "solve"
    assert payload["status"] == "satisfied"
    assert payload["solve_report"] is not None
    assert payload["table__frontier"] is not None
    materialized_dirs = sorted((run_dir / "analysis" / "materialized_hits").iterdir())
    assert len(materialized_dirs) == 2
    assert materialized_dirs[0].name == "hit_01"
    assert materialized_dirs[1].name == "hit_02"
    assert (materialized_dirs[0] / "analysis" / "views" / "post_nick_foldback.snapback_visual.v1.json").exists()
    assert (materialized_dirs[0] / "analysis" / "views" / "snapback_triptych.snapback_visual.v1.jsonl").exists()
    assert (materialized_dirs[0] / "baserender_jobs" / "snapback_triptych.job.yaml").exists()


def test_snapback_show_fails_when_solve_report_workspace_root_drifts(tmp_path: Path) -> None:
    workspace, _explicit_path, solve_path = _write_workspace(tmp_path)

    result = runner.invoke(app, ["snapback", "solve", "--spec", str(solve_path)], color=False)

    assert result.exit_code == 0
    run_dir = workspace / "outputs" / "solve"
    solve_report_path = run_dir / "analysis" / "reports" / "solve_report.json"
    solve_report = json.loads(solve_report_path.read_text(encoding="utf-8"))
    solve_report["workspace_root"] = "/tmp/drifted"
    solve_report_path.write_text(json.dumps(solve_report, indent=2), encoding="utf-8")

    show_result = runner.invoke(app, ["snapback", "show", "--run", str(run_dir)], color=False)

    assert show_result.exit_code == 1
    assert "Snapback solve report workspace_root drift detected." in show_result.output


def test_snapback_show_fails_when_materialized_hit_bundle_is_missing(tmp_path: Path) -> None:
    workspace, _explicit_path, solve_path = _write_workspace(tmp_path)

    result = runner.invoke(app, ["snapback", "solve", "--spec", str(solve_path)], color=False)

    assert result.exit_code == 0
    run_dir = workspace / "outputs" / "solve"
    shutil.rmtree(run_dir / "analysis" / "materialized_hits" / "hit_01")

    show_result = runner.invoke(app, ["snapback", "show", "--run", str(run_dir)], color=False)

    assert show_result.exit_code == 1
    assert "Materialized snapback hit bundle missing" in show_result.output


def test_snapback_show_fails_when_materialized_hit_bundle_path_is_reused(tmp_path: Path) -> None:
    workspace, _explicit_path, solve_path = _write_workspace(tmp_path)

    result = runner.invoke(app, ["snapback", "solve", "--spec", str(solve_path)], color=False)

    assert result.exit_code == 0
    run_dir = workspace / "outputs" / "solve"
    solve_report_path = run_dir / "analysis" / "reports" / "solve_report.json"
    solve_report = json.loads(solve_report_path.read_text(encoding="utf-8"))
    solve_report["hits"][0]["materialized_run_dir"] = solve_report["hits"][1]["materialized_run_dir"]
    solve_report_path.write_text(json.dumps(solve_report, indent=2), encoding="utf-8")

    show_result = runner.invoke(app, ["snapback", "show", "--run", str(run_dir)], color=False)

    assert show_result.exit_code == 1
    assert "Snapback solve materialized hit path/rank drift detected." in show_result.output


def test_snapback_show_fails_when_explicit_foldback_visual_drifts_from_report(tmp_path: Path) -> None:
    workspace, explicit_path, _solve_path = _write_workspace(tmp_path)

    design_result = runner.invoke(app, ["snapback", "design", "--spec", str(explicit_path)], color=False)

    assert design_result.exit_code == 0
    run_dir = workspace / "outputs" / "design"
    foldback_visual_path = run_dir / "analysis" / "views" / "post_nick_foldback.snapback_visual.v1.json"
    foldback_visual = json.loads(foldback_visual_path.read_text(encoding="utf-8"))
    foldback_visual["primary_sequence"] = "AAAAAAA"
    foldback_visual_path.write_text(json.dumps(foldback_visual, indent=2), encoding="utf-8")

    show_result = runner.invoke(app, ["snapback", "show", "--run", str(run_dir)], color=False)

    assert show_result.exit_code == 1
    assert "Snapback foldback visual primary_sequence drift detected." in show_result.output


def test_snapback_init_workspace_scaffolds_v2_explicit_and_v3_solve_examples(tmp_path: Path) -> None:
    target = tmp_path / "workspaces" / "demo_snapback"

    result = runner.invoke(app, ["snapback", "init-workspace", "--output", str(target)], color=False)

    assert result.exit_code == 0
    assert (target / "README.md").exists()
    assert (target / "runbook.md").exists()
    assert (target / "configs" / "runbook.yaml").exists()
    assert (target / "configs" / "snapback" / "demo_teto_bpu10i_cap.snapback.yaml").exists()
    assert (target / "configs" / "snapback" / "demo_teto_catalog_scan.snapback.solve.yaml").exists()
    assert (target / "inputs" / "nickases" / "local.nickases.yaml").exists()
    assert not (target / "snapback_workspace_manifest.json").exists()
    assert not (target / "outputs" / "design").exists()
    assert not (target / "outputs" / "solve").exists()
    solve_payload = yaml.safe_load(
        (target / "configs" / "snapback" / "demo_teto_catalog_scan.snapback.solve.yaml").read_text(encoding="utf-8")
    )
    assert solve_payload["catalog"]["preset"] == "neb_nicking_v1"
    assert solve_payload["catalog"]["additional_presets"] == ["thermo_nicking_v1"]
    assert solve_payload["catalog"]["additional_paths"] == []
    assert "goal" not in solve_payload
    assert "retained_homology_length" not in solve_payload["search"]
    assert solve_payload["search"]["min_paired_bp"] == 3
    runbook_payload = yaml.safe_load((target / "configs" / "runbook.yaml").read_text(encoding="utf-8"))
    assert runbook_payload["runbook"]["steps"][2]["run"][-1] == "outputs/design"
    assert runbook_payload["runbook"]["steps"][4]["run"][-1] == "outputs/solve"
