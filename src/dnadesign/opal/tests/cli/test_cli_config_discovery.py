"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/cli/test_cli_config_discovery.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records


def _setup_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    return workdir, campaign


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


def test_init_rejects_unknown_model_plugin(tmp_path: Path) -> None:
    _, campaign = _setup_workspace(tmp_path)
    text = campaign.read_text()
    campaign.write_text(text.replace("name: random_forest", "name: unknown_model_v99", 1))

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "init", "--config", str(campaign)])
    assert res.exit_code != 0
    assert "Unknown model plugin 'unknown_model_v99'" in res.output
