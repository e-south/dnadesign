"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/cli/test_cassette_cli.py

CLI contract tests for the cassette command group.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.baserender import app as baserender_app
from dnadesign.cruncher.cli.app import app

runner = CliRunner()
baserender_runner = CliRunner()


def _write_workspace(
    tmp_path: Path,
    *,
    emit_visual_contracts: bool = True,
    emit_baserender_jobs: bool = True,
) -> tuple[Path, Path]:
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
                    "schema_version": 1,
                    "name": "demo_hairpin",
                    "topology": {
                        "stem5p_arm": "AACGAT",
                        "loop": "TT",
                        "stem3p_arm_mode": "derive_reverse_complement",
                    },
                    "duplex_context": {"upstream": "", "downstream": ""},
                    "nicking": {
                        "designated_strand": "primary_strand",
                        "left": {"nickase": "nb_left", "nick_window": {"start": 0, "end": 3}},
                        "right": {"nickase": "nb_right", "nick_window": {"start": 11, "end": 13}},
                    },
                    "catalog": {"path": "inputs/nickases/demo.nickases.yaml"},
                    "output": {
                        "run_dir": "outputs/cassettes",
                        "emit_visual_contracts": emit_visual_contracts,
                        "emit_baserender_jobs": emit_baserender_jobs,
                        "baserender_profiles": ["duplex_qa", "hairpin_qa"],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return workspace, spec_path


def _rewrite_spec(spec_path: Path, transform) -> None:
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    transform(payload["cassette"])
    spec_path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _write_solve_workspace(tmp_path: Path, *, forbid_literal: str | None = None) -> tuple[Path, Path]:
    workspace = tmp_path / "workspaces" / "demo_cassette_solve"
    spec_path = workspace / "configs" / "cassettes" / "demo_hairpin.cassette.solve.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cassette_solve": {
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
                "forbidden_literals": [forbid_literal] if forbid_literal else [],
                "forbidden_iupac_motifs": [],
                "forbid_reverse_complements": True,
                "scope": "evaluation_context",
            },
            "sequence_quality": {},
            "catalog": {"preset": "neb_nicking_v1", "additional_paths": []},
            "search": {
                "max_hits": 3,
                "max_enumerated_candidates": 256,
                "min_pairwise_hamming_distance": 2,
                "bounded_segment_target": 10,
                "gc_target": 0.5,
                "materialize_top_k": 2,
            },
            "output": {
                "run_dir": "outputs/cassette_solves",
                "emit_visual_contracts": True,
                "emit_baserender_jobs": True,
                "baserender_profiles": [
                    "duplex_qa",
                    "hairpin_qa",
                    "top_hits_duplex_qa",
                    "top_hits_hairpin_qa",
                ],
            },
        }
    }
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return workspace, spec_path


def test_root_help_includes_cassette_group() -> None:
    result = runner.invoke(app, ["--help"], color=False)

    assert result.exit_code == 0
    assert "cassette" in result.output
    assert "dual-context hairpin cassette" in result.output


def test_cassette_help_describes_validate_design_solve_and_catalog_surface() -> None:
    result = runner.invoke(app, ["cassette", "--help"], color=False)

    assert result.exit_code == 0
    assert "init-workspace" in result.output
    assert "validate" in result.output
    assert "design" in result.output
    assert "solve" in result.output
    assert "show" in result.output
    assert "catalog" in result.output


def test_cassette_command_module_defers_workflow_import() -> None:
    command_module = "dnadesign.cruncher.cli.commands.cassette"
    workflow_module = "dnadesign.cruncher.app.cassette_workflow"
    sys.modules.pop(command_module, None)
    sys.modules.pop(workflow_module, None)

    importlib.import_module(command_module)

    assert workflow_module not in sys.modules


def test_cassette_validate_json_reports_candidate(tmp_path: Path) -> None:
    _workspace, spec_path = _write_workspace(tmp_path)

    result = runner.invoke(app, ["cassette", "validate", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "satisfied"
    assert payload["metadata"]["spec_schema_version"] == 1
    assert payload["candidate"]["intended_left_nick"]["boundary"] == 2
    assert payload["candidate"]["intended_right_nick"]["boundary"] == 12


def test_cassette_design_writes_artifacts_and_show_reads_them(tmp_path: Path) -> None:
    workspace, spec_path = _write_workspace(tmp_path)

    result = runner.invoke(app, ["cassette", "design", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 0
    run_root = workspace / "outputs" / "cassettes" / "demo_hairpin"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "meta" / "cassette_manifest.json").exists()
    assert (run_dir / "meta" / "cassette_status.json").exists()
    assert (run_dir / "analysis" / "reports" / "report.json").exists()
    assert (run_dir / "analysis" / "reports" / "report.md").exists()
    assert (run_dir / "export" / "table__candidates.csv").exists()
    assert (run_dir / "views" / "linear_duplex.v1.json").exists()
    assert (run_dir / "views" / "ssdna_hairpin.v1.json").exists()
    assert (run_dir / "views" / "views_manifest.v1.json").exists()
    assert (run_dir / "baserender_jobs" / "linear_duplex.job.yaml").exists()
    assert (run_dir / "baserender_jobs" / "ssdna_hairpin.job.yaml").exists()

    show_result = runner.invoke(app, ["cassette", "show", "--run", str(run_dir)], color=False)
    assert show_result.exit_code == 0
    assert "demo_hairpin" in show_result.output
    assert "completed" in show_result.output
    assert "Manifest ->" in show_result.output
    assert "Status file ->" in show_result.output
    assert "Views manifest" in show_result.output
    assert "Linear duplex job" in show_result.output
    assert "ssDNA hairpin job" in show_result.output


def test_cassette_design_json_is_machine_readable(tmp_path: Path) -> None:
    _workspace, spec_path = _write_workspace(tmp_path)

    result = runner.invoke(app, ["cassette", "design", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "satisfied"
    assert "/outputs/cassettes/demo_hairpin/" in payload["run_dir"]
    assert len(Path(payload["run_dir"]).name) == 12


def test_cassette_solve_json_is_machine_readable_and_materializes_hits(tmp_path: Path) -> None:
    workspace, spec_path = _write_solve_workspace(tmp_path)

    result = runner.invoke(app, ["cassette", "solve", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "solved"
    assert len(payload["hits"]) == 3
    assert payload["selection_summary"]["policy"] == "greedy_hamming"
    assert payload["selection_summary"]["selection_policy_defaulted"] is True
    run_dir = Path(payload["run_dir"])
    assert run_dir.is_dir()
    assert str(run_dir).startswith(str(workspace / "outputs" / "cassette_solves"))
    assert (run_dir / "solve_report.json").exists()
    assert (run_dir / "table__hits.csv").exists()
    assert (run_dir / "views" / "top_hits.linear_duplex.v1.jsonl").exists()
    assert (run_dir / "views" / "top_hits.ssdna_hairpin.v1.jsonl").exists()
    assert (run_dir / "baserender_jobs" / "top_hits_duplex.job.yaml").exists()
    assert (run_dir / "baserender_jobs" / "top_hits_hairpin.job.yaml").exists()
    assert len(list((run_dir / "hits").iterdir())) == 2


def test_cassette_solve_plaintext_surfaces_selection_policy_and_bounded_warnings(tmp_path: Path) -> None:
    _workspace, spec_path = _write_solve_workspace(tmp_path)
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    payload["cassette_solve"]["search"]["max_enumerated_candidates"] = 32
    payload["cassette_solve"]["search"]["materialize_top_k"] = 1
    payload["cassette_solve"]["search"]["selection"] = {
        "policy": "mmr",
        "pool_size": 8,
        "distance_metric": "hamming",
        "min_pairwise_distance": 2,
        "diversity_weight": 0.35,
    }
    payload["cassette_solve"]["search"].pop("min_pairwise_hamming_distance", None)
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["cassette", "solve", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 0
    assert "Selection -> mmr" in result.output
    assert "Selected ->" in result.output
    assert "Pool ->" in result.output
    assert "Selection bounds ->" in result.output
    assert "ACCEPTED_POOL_TRUNCATED" in result.output
    assert "SELECTION_RESULTS_SEARCH_BOUNDED" in result.output


def test_cassette_solve_plaintext_surfaces_top_hit_view_and_job_paths(tmp_path: Path) -> None:
    _workspace, spec_path = _write_solve_workspace(tmp_path)

    result = runner.invoke(app, ["cassette", "solve", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 0
    assert "Top-hit duplex views ->" in result.output
    assert "Top-hit hairpin views ->" in result.output
    assert "Top-hit duplex job ->" in result.output
    assert "Top-hit hairpin job ->" in result.output


def test_cassette_solve_no_hits_exits_nonzero_after_writing_artifacts(tmp_path: Path) -> None:
    workspace, spec_path = _write_solve_workspace(tmp_path, forbid_literal="CCTCAGC")

    result = runner.invoke(app, ["cassette", "solve", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 1
    run_root = workspace / "outputs" / "cassette_solves"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    report = json.loads((run_dir / "solve_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "no_hits"


def test_cassette_solve_invalid_catalog_exits_nonzero_after_writing_artifacts(tmp_path: Path) -> None:
    workspace, spec_path = _write_solve_workspace(tmp_path)
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    payload["cassette_solve"]["catalog"]["additional_paths"] = ["inputs/catalogs/missing.yaml"]
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["cassette", "solve", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 1
    run_root = workspace / "outputs" / "cassette_solves"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    report = json.loads((run_dir / "solve_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "invalid_catalog"


def test_cassette_init_workspace_scaffolds_runtime_profiles_and_runbook_discovery(tmp_path: Path) -> None:
    scaffold_root = tmp_path / "cassette_lab"
    sibling_workspace = tmp_path / "other_workspace"
    sibling_workspace.mkdir(parents=True, exist_ok=True)
    sibling_sentinel = sibling_workspace / "sentinel.txt"
    sibling_sentinel.write_text("keep", encoding="utf-8")

    result = runner.invoke(
        app,
        ["cassette", "init-workspace", "--output", str(scaffold_root)],
        color=False,
    )

    assert result.exit_code == 0
    assert "Cassette workspace scaffold" in result.output
    assert "runbook-only" in result.output
    assert "cruncher workspaces" in result.output
    assert "list" in result.output
    assert (scaffold_root / "README.md").exists()
    assert (scaffold_root / "runbook.md").exists()
    assert (scaffold_root / "cassette_workspace_manifest.json").exists()
    assert (scaffold_root / "configs" / "runbook.yaml").exists()
    manifest_payload = json.loads((scaffold_root / "cassette_workspace_manifest.json").read_text(encoding="utf-8"))
    assert [item["label"] for item in manifest_payload["profiles"]] == ["fast", "balanced", "deep_mmr"]
    profiles_by_filename = {item["filename"]: item for item in manifest_payload["profiles"]}
    assert profiles_by_filename["demo_hairpin_fast.cassette.solve.yaml"]["selection"]["policy"] == "greedy_hamming"
    assert profiles_by_filename["demo_hairpin_balanced.cassette.solve.yaml"]["selection"]["policy"] == "greedy_hamming"
    assert profiles_by_filename["demo_hairpin_deep_mmr.cassette.solve.yaml"]["selection"]["policy"] == "mmr"
    assert (
        profiles_by_filename["demo_hairpin_fast.cassette.solve.yaml"]["search"]["max_search_nodes"]
        < profiles_by_filename["demo_hairpin_balanced.cassette.solve.yaml"]["search"]["max_search_nodes"]
        < profiles_by_filename["demo_hairpin_deep_mmr.cassette.solve.yaml"]["search"]["max_search_nodes"]
    )
    spec_dir = scaffold_root / "configs" / "cassettes"
    profile_names = [
        "demo_hairpin_fast.cassette.solve.yaml",
        "demo_hairpin_balanced.cassette.solve.yaml",
        "demo_hairpin_deep_mmr.cassette.solve.yaml",
    ]
    for profile_name in profile_names:
        assert (spec_dir / profile_name).exists()

    for profile_name in profile_names:
        solve_result = runner.invoke(
            app,
            ["cassette", "solve", "--spec", str(spec_dir / profile_name), "--json"],
            color=False,
        )
        assert solve_result.exit_code == 0, profile_name
        payload = json.loads(solve_result.output)
        assert payload["status"] == "solved"
        profile = profiles_by_filename[profile_name]
        assert payload["selection_summary"]["policy"] == profile["selection"]["policy"]
        assert payload["selection_summary"]["pool_size"] == profile["selection"]["pool_size"]
        assert payload["selection_summary"]["selected_hit_count"] <= profile["search"]["max_hits"]
        assert payload["metadata"]["enumerated_candidate_count"] <= profile["search"]["max_enumerated_candidates"]
        assert str(payload["run_dir"]).startswith(str(scaffold_root / "outputs" / "cassette_solves"))

    list_result = runner.invoke(
        app,
        ["workspaces", "list", "--root", str(tmp_path)],
        env={"CRUNCHER_NONINTERACTIVE": "1", "COLUMNS": "240"},
        color=False,
    )
    assert list_result.exit_code == 0
    assert "cassette_lab" in list_result.output
    assert "runbook-only" in list_result.output

    assert sibling_sentinel.read_text(encoding="utf-8") == "keep"
    assert not (sibling_workspace / "outputs" / "cassette_solves").exists()


def test_cassette_init_workspace_supports_workspace_name_plus_root(tmp_path: Path) -> None:
    workspaces_root = tmp_path / "workspaces"

    result = runner.invoke(
        app,
        ["cassette", "init-workspace", "demo_cassette", "--root", str(workspaces_root)],
        env={"COLUMNS": "60"},
        color=False,
    )

    scaffold_root = workspaces_root / "demo_cassette"
    assert result.exit_code == 0
    assert "demo_cassette" in result.output
    assert scaffold_root.is_dir()
    assert (scaffold_root / "configs" / "runbook.yaml").exists()

    list_result = runner.invoke(
        app,
        ["workspaces", "list", "--root", str(workspaces_root)],
        env={"CRUNCHER_NONINTERACTIVE": "1", "COLUMNS": "240"},
        color=False,
    )
    assert list_result.exit_code == 0
    assert "demo_cassette" in list_result.output
    assert "runbook-only" in list_result.output


def test_cassette_init_workspace_solve_and_baserender_cli_render_in_place(tmp_path: Path) -> None:
    scaffold_root = tmp_path / "cassette_lab"

    init_result = runner.invoke(
        app,
        ["cassette", "init-workspace", "--output", str(scaffold_root)],
        color=False,
    )

    assert init_result.exit_code == 0
    solve_spec = scaffold_root / "configs" / "cassettes" / "demo_hairpin_fast.cassette.solve.yaml"
    solve_result = runner.invoke(
        app,
        ["cassette", "solve", "--spec", str(solve_spec), "--json"],
        color=False,
    )

    assert solve_result.exit_code == 0
    payload = json.loads(solve_result.output)
    run_dir = Path(payload["run_dir"])
    assert run_dir.is_dir()
    assert str(run_dir).startswith(str(scaffold_root / "outputs" / "cassette_solves"))

    solve_job = run_dir / "baserender_jobs" / "top_hits_duplex.job.yaml"
    solve_validate = baserender_runner.invoke(
        baserender_app,
        ["job", "validate", str(solve_job)],
        color=False,
    )
    assert solve_validate.exit_code == 0
    solve_run = baserender_runner.invoke(
        baserender_app,
        ["job", "run", str(solve_job)],
        color=False,
    )
    assert solve_run.exit_code == 0
    solve_render = run_dir / "renders" / "top_hits_duplex_qa_sheet.pdf"
    assert solve_render.exists()
    solve_render.resolve().relative_to(scaffold_root.resolve())

    first_hit_dir = Path(payload["hits"][0]["materialized_run_dir"])
    hairpin_job = first_hit_dir / "baserender_jobs" / "ssdna_hairpin.job.yaml"
    hairpin_validate = baserender_runner.invoke(
        baserender_app,
        ["job", "validate", str(hairpin_job)],
        color=False,
    )
    assert hairpin_validate.exit_code == 0
    hairpin_run = baserender_runner.invoke(
        baserender_app,
        ["job", "run", str(hairpin_job)],
        color=False,
    )
    assert hairpin_run.exit_code == 0
    hit_render = first_hit_dir / "renders" / "ssdna_hairpin.pdf"
    assert hit_render.exists()
    hit_render.resolve().relative_to(scaffold_root.resolve())


def test_cassette_solve_plaintext_surfaces_policy_underfill_warning(tmp_path: Path) -> None:
    _workspace, spec_path = _write_solve_workspace(tmp_path)
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    payload["cassette_solve"]["search"]["max_hits"] = 5
    payload["cassette_solve"]["search"]["max_enumerated_candidates"] = 20000
    payload["cassette_solve"]["search"]["max_search_nodes"] = 500000
    payload["cassette_solve"]["search"]["selection"] = {
        "policy": "greedy_hamming",
        "pool_size": 1024,
        "distance_metric": "hamming",
        "min_pairwise_distance": 9,
    }
    payload["cassette_solve"]["search"].pop("min_pairwise_hamming_distance", None)
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["cassette", "solve", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 0
    assert "Selection filter -> selection_policy_constraints_filtered_pool" in result.output
    assert "Warning -> SELECTION_POLICY_LIMITED_HITS" in result.output


def test_cassette_init_workspace_refuses_to_overwrite_nonempty_unowned_root(tmp_path: Path) -> None:
    workspace_root = tmp_path / "cassette_lab"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "notes.txt").write_text("user data", encoding="utf-8")

    result = runner.invoke(
        app,
        ["cassette", "init-workspace", "--output", str(workspace_root), "--force-overwrite"],
        color=False,
    )

    assert result.exit_code == 1
    assert "Refusing to overwrite" in result.output
    assert (workspace_root / "notes.txt").read_text(encoding="utf-8") == "user data"


def test_cassette_init_workspace_rejects_symlink_root(tmp_path: Path) -> None:
    external_root = tmp_path / "external_root"
    external_root.mkdir(parents=True, exist_ok=True)
    symlink_root = tmp_path / "cassette_link"
    symlink_root.symlink_to(external_root, target_is_directory=True)

    result = runner.invoke(
        app,
        ["cassette", "init-workspace", "--output", str(symlink_root)],
        color=False,
    )

    assert result.exit_code == 1
    assert "must not be a symlink" in result.output
    assert not (external_root / "README.md").exists()


def test_cassette_init_workspace_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    external_root = tmp_path / "external_root"
    external_root.mkdir(parents=True, exist_ok=True)
    symlink_parent = tmp_path / "workspace_alias"
    symlink_parent.symlink_to(external_root, target_is_directory=True)

    result = runner.invoke(
        app,
        ["cassette", "init-workspace", "--output", str(symlink_parent / "cassette_lab")],
        color=False,
    )

    assert result.exit_code == 1
    assert "must not traverse a symlinked directory" in result.output
    assert not (external_root / "cassette_lab").exists()


def test_cassette_catalog_init_neb_writes_builtin_preset(tmp_path: Path) -> None:
    output_path = tmp_path / "configs" / "catalogs" / "neb_nicking_v1.yaml"

    result = runner.invoke(app, ["cassette", "catalog", "init-neb", "--output", str(output_path)], color=False)

    assert result.exit_code == 0
    assert output_path.exists()
    payload = output_path.read_text(encoding="utf-8")
    assert "preset_id: neb_nicking_v1" in payload
    assert "WarmStart Nt.BstNBI" in payload


def test_cassette_design_respects_visual_contract_toggle(tmp_path: Path) -> None:
    workspace, spec_path = _write_workspace(
        tmp_path,
        emit_visual_contracts=False,
        emit_baserender_jobs=False,
    )

    result = runner.invoke(app, ["cassette", "design", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 0
    run_root = workspace / "outputs" / "cassettes" / "demo_hairpin"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert not (run_dir / "views" / "linear_duplex.v1.json").exists()
    assert not (run_dir / "views" / "ssdna_hairpin.v1.json").exists()
    assert not (run_dir / "views" / "views_manifest.v1.json").exists()
    assert not (run_dir / "baserender_jobs" / "linear_duplex.job.yaml").exists()
    assert not (run_dir / "baserender_jobs" / "ssdna_hairpin.job.yaml").exists()

    show_result = runner.invoke(app, ["cassette", "show", "--run", str(run_dir)], color=False)
    assert show_result.exit_code == 0
    assert "Views manifest" not in show_result.output
    assert "Linear duplex job" not in show_result.output


def test_cassette_validate_plaintext_reports_structured_issue_codes(tmp_path: Path) -> None:
    _workspace, spec_path = _write_workspace(tmp_path)

    def _mutate(cassette: dict[str, object]) -> None:
        nicking = cassette["nicking"]
        assert isinstance(nicking, dict)
        right = nicking["right"]
        assert isinstance(right, dict)
        right["nick_window"] = {"start": 13, "end": 13}

    _rewrite_spec(spec_path, _mutate)

    result = runner.invoke(app, ["cassette", "validate", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 1
    assert "Mode -> schema v1 / legacy_v1" in result.output
    assert "RIGHT_WINDOW_NO_MATCH" in result.output


def test_cassette_design_writes_unsatisfied_artifacts_before_exit(tmp_path: Path) -> None:
    workspace, spec_path = _write_workspace(tmp_path)

    def _mutate(cassette: dict[str, object]) -> None:
        nicking = cassette["nicking"]
        assert isinstance(nicking, dict)
        right = nicking["right"]
        assert isinstance(right, dict)
        right["nick_window"] = {"start": 13, "end": 13}

    _rewrite_spec(spec_path, _mutate)

    result = runner.invoke(app, ["cassette", "design", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 1
    run_root = workspace / "outputs" / "cassettes" / "demo_hairpin"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    report = json.loads((run_dir / "analysis" / "reports" / "report.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "meta" / "cassette_status.json").read_text(encoding="utf-8"))

    assert report["status"] == "unsatisfied"
    assert report["issues"][0]["code"] == "RIGHT_WINDOW_NO_MATCH"
    assert status["status"] == "unsatisfied"
    assert "legacy_v1" in status["status_message"]
    assert not (run_dir / "views" / "linear_duplex.v1.json").exists()
    assert not (run_dir / "views" / "ssdna_hairpin.v1.json").exists()
    assert not (run_dir / "views" / "views_manifest.v1.json").exists()
    assert not (run_dir / "baserender_jobs" / "linear_duplex.job.yaml").exists()
    assert not (run_dir / "baserender_jobs" / "ssdna_hairpin.job.yaml").exists()


def test_cassette_design_writes_only_to_selected_workspace_root(tmp_path: Path) -> None:
    workspace_a, spec_path_a = _write_workspace(tmp_path / "a")
    workspace_b, _spec_path_b = _write_workspace(tmp_path / "b")

    result = runner.invoke(app, ["cassette", "design", "--spec", str(spec_path_a)], color=False)

    assert result.exit_code == 0
    assert (workspace_a / "outputs" / "cassettes" / "demo_hairpin").exists()
    assert not (workspace_b / "outputs" / "cassettes" / "demo_hairpin").exists()


def test_existing_help_commands_still_work() -> None:
    for args in (["sample", "--help"], ["study", "--help"], ["portfolio", "--help"]):
        result = runner.invoke(app, args, color=False)
        assert result.exit_code == 0, args
