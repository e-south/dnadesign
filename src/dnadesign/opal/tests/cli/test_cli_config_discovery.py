"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_config_discovery.py

Regression tests for CLI config discovery OPAL CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.cli.commands.notebook_support import marimo_subprocess_environment
from dnadesign.opal.src.reporting.notebook_set import build_campaign_set_notebook_view_model
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records


def _setup_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    return workdir, campaign


def _setup_usr_workspace(
    tmp_path: Path,
    *,
    plots: list[dict[str, str]] | None = None,
) -> tuple[Path, Path, Path]:
    workdir = tmp_path / "campaign"
    workdir.mkdir()
    operator_root = tmp_path / "operator-data"
    dataset_root = operator_root / "demo_candidates"
    dataset_root.mkdir(parents=True)
    records = dataset_root / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records, plots=plots)
    payload = yaml.safe_load(campaign.read_text(encoding="utf-8"))
    payload["ownership"] = {
        "owner_scope": "study_campaign",
        "study_id": "demo_study",
        "dataset_id": "demo_candidates",
        "portable": False,
    }
    payload["data"]["location"] = {
        "kind": "usr",
        "path": str(tmp_path / "stale-root"),
        "dataset": "demo_candidates",
    }
    campaign.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return operator_root, campaign, records


def test_config_discovery_env_var(monkeypatch, tmp_path: Path) -> None:
    _, campaign = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    monkeypatch.setenv("OPAL_CONFIG", str(campaign))
    monkeypatch.chdir(tmp_path)

    res = runner.invoke(app, ["--no-color", "validate"])
    assert res.exit_code == 0, res.output


def test_config_discovery_marker_relative_to_workdir_is_ignored(monkeypatch, tmp_path: Path) -> None:
    workdir, _ = _setup_workspace(tmp_path)
    marker_dir = workdir / ".opal"
    marker_dir.mkdir(parents=True, exist_ok=True)
    # marker paths resolve relative to workdir
    (marker_dir / "config").write_text("campaign.yaml")

    sub = workdir / "nested"
    sub.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(sub)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "validate"])
    assert res.exit_code != 0
    assert "No config provided" in res.output


def test_config_discovery_env_invalid_errors(monkeypatch, tmp_path: Path) -> None:
    _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    bad = tmp_path / "missing.yaml"
    monkeypatch.setenv("OPAL_CONFIG", str(bad))
    monkeypatch.chdir(tmp_path)

    res = runner.invoke(app, ["--no-color", "validate"])
    assert res.exit_code != 0
    assert "OPAL_CONFIG points to a missing path" in res.output


def test_config_directory_rejected(tmp_path: Path) -> None:
    workdir, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "validate", "--config", str(workdir)])
    assert res.exit_code != 0
    assert "Config path is a directory" in res.output


def test_config_required_without_flag_or_env(monkeypatch, tmp_path: Path) -> None:
    workdir, _ = _setup_workspace(tmp_path)
    monkeypatch.chdir(workdir)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "validate"])
    assert res.exit_code != 0
    assert "No config provided" in res.output


def test_config_required_ignores_marker(monkeypatch, tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    marker_dir = workdir / ".opal"
    marker_dir.mkdir(parents=True, exist_ok=True)
    (marker_dir / "config").write_text(str(campaign))
    monkeypatch.chdir(workdir)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "validate"])
    assert res.exit_code != 0
    assert "No config provided" in res.output


def test_config_discovery_explicit_flag(monkeypatch, tmp_path: Path) -> None:
    _, campaign = _setup_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "validate", "--config", str(campaign)])
    assert res.exit_code == 0, res.output


def test_usr_root_is_an_explicit_global_cli_coordinate(tmp_path: Path) -> None:
    operator_root = tmp_path / "operator-data"
    operator_root.mkdir()
    _, campaign = _setup_workspace(tmp_path)

    result = CliRunner().invoke(
        _build(),
        ["--no-color", "--usr-root", str(operator_root), "validate", "--config", str(campaign)],
    )

    assert result.exit_code != 0
    assert "only valid when data.location.kind=usr" in result.output


def test_usr_root_reaches_plot_campaign_analysis(tmp_path: Path) -> None:
    operator_root, campaign, records = _setup_usr_workspace(
        tmp_path,
        plots=[{"name": "objective", "kind": "objective_scatter"}],
    )

    result = CliRunner().invoke(
        _build(),
        [
            "--no-color",
            "--usr-root",
            str(operator_root),
            "plot",
            "--config",
            str(campaign),
            "--list-config",
        ],
    )

    assert result.exit_code == 0, result.output
    assert str(records) in result.output
    assert "stale-root" not in result.output


def test_usr_root_reaches_notebook_campaign_analysis(tmp_path: Path) -> None:
    operator_root, campaign, records = _setup_usr_workspace(tmp_path)
    notebook = tmp_path / "campaign_notebook.py"

    result = CliRunner().invoke(
        _build(),
        [
            "--no-color",
            "--usr-root",
            str(operator_root),
            "notebook",
            "generate",
            "--config",
            str(campaign),
            "--out",
            str(notebook),
            "--no-validate",
        ],
    )

    assert result.exit_code == 0, result.output
    assert notebook.is_file()
    text = notebook.read_text(encoding="utf-8")
    usr_root_assignment = next(
        node
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "usr_root" for target in node.targets)
    )
    assert isinstance(usr_root_assignment.value, ast.Call)
    assert ast.literal_eval(usr_root_assignment.value.args[0]) == str(operator_root)
    assert "usr_root=usr_root" in text
    runtime_model = build_campaign_set_notebook_view_model(
        [campaign],
        round_selector="latest",
        usr_root=operator_root,
    )
    assert runtime_model["campaigns"][0]["campaign"]["records_path"] == str(records)
    assert runtime_model["campaigns"][0]["campaign"]["usr_root"] == str(operator_root)


def test_marimo_subprocess_environment_attests_exact_usr_root(tmp_path: Path) -> None:
    operator_root = tmp_path / "operator-data"
    operator_root.mkdir()

    environment = marimo_subprocess_environment(operator_root, base_environment={"PATH": "/bin"})

    assert environment == {
        "PATH": "/bin",
        "OPAL_NOTEBOOK_USR_ROOT": str(operator_root.resolve()),
    }


def test_init_rejects_unknown_model_plugin(tmp_path: Path) -> None:
    _, campaign = _setup_workspace(tmp_path)
    text = campaign.read_text()
    campaign.write_text(text.replace("name: random_forest", "name: unknown_model_v99", 1))

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "init", "--config", str(campaign)])
    assert res.exit_code != 0
    assert "Unknown model plugin 'unknown_model_v99'" in res.output
