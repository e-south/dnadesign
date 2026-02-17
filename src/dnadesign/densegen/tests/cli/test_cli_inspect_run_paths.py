"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_cli_inspect_run_paths.py

CLI tests for workspace-relative path rendering in inspect run.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.densegen.src.cli.main import app
from dnadesign.densegen.tests.config_fixtures import write_minimal_config

PLAN_POOL_LABEL = "plan_pool__demo_plan"


def test_inspect_run_uses_relative_root(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\nTF1,AAA\n")
    meta_root = tmp_path / "outputs" / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "run_id": "demo",
        "created_at": "2026-01-14T00:00:00+00:00",
        "schema_version": "2.9",
        "config_sha256": "dummy",
        "run_root": str(tmp_path),
        "random_seed": 0,
        "seed_stage_a": 0,
        "seed_stage_b": 0,
        "seed_solver": 0,
        "solver_backend": "CBC",
        "solver_strategy": "iterate",
        "solver_time_limit_seconds": None,
        "solver_threads": None,
        "solver_strands": "double",
        "dense_arrays_version": None,
        "dense_arrays_version_source": "unknown",
        "items": [
            {
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "generated": 0,
                "duplicates_skipped": 0,
                "failed_solutions": 0,
                "total_resamples": 0,
                "libraries_built": 0,
                "stall_events": 0,
            }
        ],
    }
    (meta_root / "run_manifest.json").write_text(json.dumps(run_manifest))

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "run", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert str(tmp_path) not in result.output
    assert "Root: ." in result.output


def test_inspect_run_root_listing_uses_relative_config_paths(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces"
    workspace_root.mkdir()
    run_dir = workspace_root / "demo_run"
    run_dir.mkdir()
    cfg_path = run_dir / "config.yaml"
    write_minimal_config(cfg_path)
    (run_dir / "inputs.csv").write_text("tf,tfbs\nTF1,AAA\n")

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "run", "--root", str(workspace_root)])
    assert result.exit_code == 0, result.output
    assert str(workspace_root) not in result.output
    assert "demo_run/config.yaml" in result.output


def test_inspect_config_missing_uses_relative_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "missing_config.yaml"
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "config", "-c", str(cfg_path)])
    assert result.exit_code != 0
    assert str(cfg_path) not in result.output
    assert "Config file not found" in result.output


def test_inspect_run_usr_events_path_prints_absolute_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [usr]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                usr:
                  root: outputs/usr_datasets
                  dataset: densegen/demo
                  chunk_size: 16
              generation:
                sequence_length: 10
                plan:
                  - name: demo_plan
                    quota: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "inputs.csv").write_text("tf,tfbs\nTF1,AAA\n")

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "run", "-c", str(cfg_path), "--usr-events-path"])

    expected = tmp_path / "outputs" / "usr_datasets" / "densegen" / "demo" / ".events.log"
    assert result.exit_code == 0, result.output
    assert str(expected.resolve()) in result.output.strip()
    assert "Run:" not in result.output


def test_inspect_run_usr_events_path_requires_usr_target(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\nTF1,AAA\n")

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "run", "-c", str(cfg_path), "--usr-events-path"])

    assert result.exit_code == 1
    assert "output.targets must include 'usr'" in result.output
