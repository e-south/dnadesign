"""
--------------------------------------------------------------------------------
<cruncher project>
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

from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def _write_workspace(tmp_path: Path, *, write_render_contract: bool = True) -> tuple[Path, Path]:
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
                        "write_render_contract": write_render_contract,
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
                "bounded_segment_target": 24,
                "gc_target": 0.5,
                "materialize_top_k": 2,
            },
            "output": {
                "run_dir": "outputs/cassette_solves",
                "write_render_contract": True,
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
    assert (run_dir / "analysis" / "reports" / "render_contract.json").exists()
    assert (run_dir / "export" / "table__candidates.csv").exists()

    show_result = runner.invoke(app, ["cassette", "show", "--run", str(run_dir)], color=False)
    assert show_result.exit_code == 0
    assert "demo_hairpin" in show_result.output
    assert "completed" in show_result.output


def test_cassette_design_json_is_machine_readable(tmp_path: Path) -> None:
    _workspace, spec_path = _write_workspace(tmp_path)

    result = runner.invoke(app, ["cassette", "design", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "satisfied"
    assert payload["run_dir"].endswith("/demo_hairpin/c8011c470b18")


def test_cassette_solve_json_is_machine_readable_and_materializes_hits(tmp_path: Path) -> None:
    workspace, spec_path = _write_solve_workspace(tmp_path)

    result = runner.invoke(app, ["cassette", "solve", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "solved"
    assert len(payload["hits"]) == 3
    run_dir = Path(payload["run_dir"])
    assert run_dir.is_dir()
    assert str(run_dir).startswith(str(workspace / "outputs" / "cassette_solves"))
    assert (run_dir / "solve_report.json").exists()
    assert (run_dir / "table__hits.csv").exists()
    assert len(list((run_dir / "hits").iterdir())) == 2


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


def test_cassette_catalog_init_neb_writes_builtin_preset(tmp_path: Path) -> None:
    output_path = tmp_path / "configs" / "catalogs" / "neb_nicking_v1.yaml"

    result = runner.invoke(app, ["cassette", "catalog", "init-neb", "--output", str(output_path)], color=False)

    assert result.exit_code == 0
    assert output_path.exists()
    payload = output_path.read_text(encoding="utf-8")
    assert "preset_id: neb_nicking_v1" in payload
    assert "WarmStart Nt.BstNBI" in payload


def test_cassette_design_respects_render_contract_toggle(tmp_path: Path) -> None:
    workspace, spec_path = _write_workspace(tmp_path, write_render_contract=False)

    result = runner.invoke(app, ["cassette", "design", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 0
    run_root = workspace / "outputs" / "cassettes" / "demo_hairpin"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert not (run_dir / "analysis" / "reports" / "render_contract.json").exists()

    show_result = runner.invoke(app, ["cassette", "show", "--run", str(run_dir)], color=False)
    assert show_result.exit_code == 0
    assert "Render contract" not in show_result.output


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


def test_existing_help_commands_still_work() -> None:
    for args in (["sample", "--help"], ["study", "--help"], ["portfolio", "--help"]):
        result = runner.invoke(app, args, color=False)
        assert result.exit_code == 0, args
