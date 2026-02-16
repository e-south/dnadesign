"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_cli_workspace_init.py

Workspace init and Stage-B guardrail tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.densegen.src.cli import app
from dnadesign.densegen.src.cli_commands import workspace as workspace_commands


def _write_source_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo
                  type: binding_sites
                  path: inputs/sites.csv
            """
        ).strip()
        + "\n"
    )


def _write_min_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo
                  type: binding_sites
                  path: inputs.csv

              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/records.parquet

              generation:
                sequence_length: 10
                plan:
                  - name: default
                    quota: 1
                    sampling:
                      include_inputs: [demo]
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


def test_workspace_init_warns_on_relative_inputs_without_copy(tmp_path: Path) -> None:
    source_path = tmp_path / "source.yaml"
    _write_source_config(source_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "workspace",
            "init",
            "--id",
            "demo_run",
            "--root",
            str(tmp_path),
            "--from-config",
            str(source_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Workspace uses file-based inputs with relative paths" in result.output
    assert (tmp_path / "demo_run" / "config.yaml").exists()


def test_stage_b_reports_missing_pool_manifest(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_min_config(cfg_path)
    pool_dir = tmp_path / "outputs" / "pools"
    pool_dir.mkdir(parents=True)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "stage-b",
            "build-libraries",
            "-c",
            str(cfg_path),
            "--pool",
            str(pool_dir),
        ],
    )
    assert result.exit_code != 0, result.output
    assert "Pool manifest not found" in result.output
    normalized = " ".join(result.output.split())
    assert "dense stage-a build-pool" in normalized


def test_workspace_init_supports_binding_sites_demo_workspace(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "workspace",
            "init",
            "--id",
            "demo_run",
            "--root",
            str(tmp_path),
            "--from-workspace",
            "demo_tfbs_baseline",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "demo_run" / "config.yaml").exists()
    assert not (tmp_path / "demo_run" / "outputs" / "report").exists()


def test_workspace_init_rejects_archived_source_workspace(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "workspace",
            "init",
            "--id",
            "demo_run",
            "--root",
            str(tmp_path),
            "--from-workspace",
            "archived",
        ],
    )
    assert result.exit_code == 1
    assert "Unknown source workspace" in result.output


def test_workspace_init_uses_env_workspace_root_when_root_not_provided(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace_root = tmp_path / "central_workspaces"
    monkeypatch.setenv("DENSEGEN_WORKSPACE_ROOT", str(workspace_root))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "workspace",
            "init",
            "--id",
            "demo_run",
            "--from-workspace",
            "demo_tfbs_baseline",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (workspace_root / "demo_run" / "config.yaml").exists()


def test_workspace_init_output_mode_usr_sets_usr_target(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "workspace",
            "init",
            "--id",
            "demo_run",
            "--root",
            str(tmp_path),
            "--from-workspace",
            "demo_tfbs_baseline",
            "--output-mode",
            "usr",
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = yaml.safe_load((tmp_path / "demo_run" / "config.yaml").read_text())
    output = cfg["densegen"]["output"]
    assert output["targets"] == ["usr"]
    assert output["usr"]["root"] == "outputs/usr_datasets"
    assert output["usr"]["dataset"] == "demo_run"
    assert (tmp_path / "demo_run" / "outputs" / "usr_datasets" / "registry.yaml").exists()


def test_workspace_init_output_mode_both_sets_both_targets(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "workspace",
            "init",
            "--id",
            "demo_run",
            "--root",
            str(tmp_path),
            "--from-workspace",
            "demo_tfbs_baseline",
            "--output-mode",
            "both",
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = yaml.safe_load((tmp_path / "demo_run" / "config.yaml").read_text())
    output = cfg["densegen"]["output"]
    assert set(output["targets"]) == {"parquet", "usr"}
    assert output["parquet"]["path"] == "outputs/tables/records.parquet"
    assert output["usr"]["root"] == "outputs/usr_datasets"
    assert (tmp_path / "demo_run" / "outputs" / "usr_datasets" / "registry.yaml").exists()


def test_workspace_init_existing_workspace_dir_shows_actionable_error(tmp_path: Path) -> None:
    existing = tmp_path / "demo_run"
    existing.mkdir(parents=True, exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "workspace",
            "init",
            "--id",
            "demo_run",
            "--root",
            str(tmp_path),
            "--from-workspace",
            "demo_tfbs_baseline",
        ],
    )

    assert result.exit_code != 0
    assert "Workspace directory already exists" in result.output
    assert "Choose a new --id or remove the existing workspace directory" in result.output


def test_workspace_where_json_reports_roots(tmp_path: Path, monkeypatch) -> None:
    workspace_root = tmp_path / "flat_runs"
    monkeypatch.setenv("DENSEGEN_WORKSPACE_ROOT", str(workspace_root))
    runner = CliRunner()
    result = runner.invoke(app, ["workspace", "where", "--format", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["workspace_root"] == str(workspace_root)
    assert payload["workspace_root_source"] == "env:DENSEGEN_WORKSPACE_ROOT"
    assert payload["workspace_source_root"].endswith("src/dnadesign/densegen/workspaces")


def test_workspace_where_json_reports_repo_workspace_root_not_runs(monkeypatch) -> None:
    repo_root = Path("/tmp/repo")
    monkeypatch.delenv("DENSEGEN_WORKSPACE_ROOT", raising=False)
    monkeypatch.setattr(workspace_commands, "_repo_root_from", lambda _start: repo_root)

    runner = CliRunner()
    result = runner.invoke(app, ["workspace", "where", "--format", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["workspace_root"] == str(repo_root / "src" / "dnadesign" / "densegen" / "workspaces")


def test_workspace_where_requires_explicit_root_outside_repo(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DENSEGEN_WORKSPACE_ROOT", raising=False)
    monkeypatch.setattr(workspace_commands, "_repo_root_from", lambda _start: None)
    runner = CliRunner()
    result = runner.invoke(app, ["workspace", "where"])
    assert result.exit_code == 1
    assert "Unable to determine workspace root" in result.output
    assert "DENSEGEN_WORKSPACE_ROOT" in result.output
