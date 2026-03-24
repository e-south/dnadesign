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
    assert payload["candidate"]["left_nick"]["nick_coordinate"] == 2
    assert payload["candidate"]["right_nick"]["nick_coordinate"] == 12


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


def test_existing_help_commands_still_work() -> None:
    for args in (["sample", "--help"], ["study", "--help"], ["portfolio", "--help"]):
        result = runner.invoke(app, args, color=False)
        assert result.exit_code == 0, args
