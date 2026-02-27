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

import pandas as pd
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
        "solver_attempt_timeout_seconds": None,
        "solver_threads": None,
        "solver_strands": "double",
        "dense_arrays_version": None,
        "dense_arrays_version_source": "unknown",
        "total_quota": 5,
        "items": [
            {
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "quota": 5,
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


def test_inspect_run_reports_quota_progress(tmp_path: Path) -> None:
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
        "solver_attempt_timeout_seconds": None,
        "solver_threads": None,
        "solver_strands": "double",
        "dense_arrays_version": None,
        "dense_arrays_version_source": "unknown",
        "total_quota": 5,
        "items": [
            {
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "quota": 5,
                "generated": 4,
                "duplicates_skipped": 0,
                "failed_solutions": 0,
                "total_resamples": 0,
                "libraries_built": 1,
                "stall_events": 0,
            }
        ],
    }
    (meta_root / "run_manifest.json").write_text(json.dumps(run_manifest))

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "run", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "4/5 (80.00%)" in result.output
    assert "Quota:" in result.output


def test_inspect_run_shows_explicit_failure_outcome_paths(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\nTF1,AAA\n")
    meta_root = tmp_path / "outputs" / "meta"
    tables_root = tmp_path / "outputs" / "tables"
    meta_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
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
        "solver_attempt_timeout_seconds": None,
        "solver_threads": None,
        "solver_strands": "double",
        "dense_arrays_version": None,
        "dense_arrays_version_source": "unknown",
        "total_quota": 5,
        "items": [
            {
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "quota": 5,
                "generated": 0,
                "duplicates_skipped": 0,
                "failed_solutions": 2,
                "total_resamples": 2,
                "libraries_built": 2,
                "stall_events": 1,
            }
        ],
    }
    (meta_root / "run_manifest.json").write_text(json.dumps(run_manifest))
    attempts_df = pd.DataFrame(
        [
            {
                "attempt_id": "a1",
                "attempt_index": 1,
                "run_id": "demo",
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "created_at": "2026-01-14T00:00:01+00:00",
                "status": "failed",
                "reason": "no_solution",
                "detail_json": "{}",
                "sequence": "",
                "sequence_hash": "",
                "solution_id": "",
                "used_tf_counts_json": "{}",
                "used_tf_list": [],
                "sampling_library_index": 1,
                "sampling_library_hash": "h1",
                "solver_status": "no_solution",
                "solver_objective": None,
                "solver_solve_time_s": 4.5,
                "dense_arrays_version": None,
                "dense_arrays_version_source": "unknown",
                "library_tfbs": ["AAA", "CCC"],
                "library_tfs": ["TF1", "TF1"],
                "library_site_ids": ["s1", "s2"],
                "library_sources": ["demo", "demo"],
            },
            {
                "attempt_id": "a2",
                "attempt_index": 2,
                "run_id": "demo",
                "input_name": PLAN_POOL_LABEL,
                "plan_name": "demo_plan",
                "created_at": "2026-01-14T00:00:02+00:00",
                "status": "failed",
                "reason": "stall_no_solution",
                "detail_json": '{"stall_seconds": 10}',
                "sequence": "",
                "sequence_hash": "",
                "solution_id": "",
                "used_tf_counts_json": "{}",
                "used_tf_list": [],
                "sampling_library_index": 2,
                "sampling_library_hash": "h2",
                "solver_status": "stall_no_solution",
                "solver_objective": None,
                "solver_solve_time_s": 4.9,
                "dense_arrays_version": None,
                "dense_arrays_version_source": "unknown",
                "library_tfbs": ["AAA", "CCC"],
                "library_tfs": ["TF1", "TF1"],
                "library_site_ids": ["s1", "s2"],
                "library_sources": ["demo", "demo"],
            },
        ]
    )
    attempts_df.to_parquet(tables_root / "attempts.parquet", index=False)

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "run", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "Failure outcomes" in result.output
    assert "no_solution" in result.output
    assert "stall_no_solution" in result.output
    assert "solver search exhausted with no accepted solution" in result.output
    assert "stalled before any accepted solution and triggered resample" in result.output


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


def test_inspect_config_missing_uses_relative_path(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "missing_config.yaml"
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "config", "-c", cfg_path.name])
    assert result.exit_code != 0
    assert "Config file not found:" in result.output
    shown_path = result.output.split("Config file not found:", 1)[1].replace("\n", "").strip()
    assert shown_path
    assert not Path(shown_path).is_absolute()
    assert shown_path.endswith("missing_config.yaml")


def test_inspect_config_missing_absolute_path_stays_absolute(tmp_path: Path) -> None:
    cfg_path = (tmp_path / "missing_absolute_config.yaml").resolve()
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "config", "-c", str(cfg_path)])
    assert result.exit_code != 0
    assert "Config file not found:" in result.output
    shown_path = result.output.split("Config file not found:", 1)[1].replace("\n", "").strip()
    assert shown_path
    assert Path(shown_path).is_absolute()
    assert shown_path == str(cfg_path)


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
                    sequences: 1
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
