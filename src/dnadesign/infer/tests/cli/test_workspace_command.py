"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/cli/test_workspace_command.py

Workspace command contracts for infer CLI ergonomics and preflight scaffolding.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.infer.cli import app

_RUNNER = CliRunner()


def test_workspace_where_uses_env_root_when_set(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    monkeypatch.setenv("INFER_WORKSPACE_ROOT", root.as_posix())

    result = _RUNNER.invoke(app, ["workspace", "where"])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert f"workspace_root: {root.resolve()}" in output
    assert "workspace_root_source: env" in output


def test_workspace_init_creates_default_layout_and_config(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"

    result = _RUNNER.invoke(app, ["workspace", "init", "--id", "demo_pressure", "--root", root.as_posix()])

    workspace_dir = root / "demo_pressure"
    assert result.exit_code == 0, result.stdout
    config_path = workspace_dir / "config.yaml"
    assert config_path.is_file()
    assert (workspace_dir / "inputs").is_dir()
    assert (workspace_dir / "outputs" / "logs" / "ops" / "audit").is_dir()
    config = config_path.read_text(encoding="utf-8")
    assert "source: records" in config
    assert "path: inputs/records.jsonl" in config
    output = result.stdout or ""
    assert "infer validate config --config" in output
    assert "infer run --config" in output


def test_workspace_init_usr_pressure_profile_uses_usr_template(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"

    result = _RUNNER.invoke(
        app,
        [
            "workspace",
            "init",
            "--id",
            "demo_pressure_usr",
            "--root",
            root.as_posix(),
            "--profile",
            "usr-pressure",
        ],
    )

    workspace_dir = root / "demo_pressure_usr"
    assert result.exit_code == 0, result.stdout
    config = (workspace_dir / "config.yaml").read_text(encoding="utf-8")
    assert "source: usr" in config
    assert "dataset: test_stress_ethanol" in config
    output = result.stdout or ""
    assert "profile: usr-pressure" in output
    assert "Review ingest.dataset and ingest.root in config.yaml before running." in output


def test_workspace_init_rejects_unknown_profile(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"

    result = _RUNNER.invoke(
        app,
        [
            "workspace",
            "init",
            "--id",
            "demo_pressure_usr",
            "--root",
            root.as_posix(),
            "--profile",
            "unknown",
        ],
    )

    assert result.exit_code == 2
    assert "workspace profile must be one of" in (result.stdout or "")


def test_workspace_local_profile_supports_validate_and_dry_run(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    workspace_id = "demo_local_flow"

    init_result = _RUNNER.invoke(
        app,
        ["workspace", "init", "--id", workspace_id, "--root", root.as_posix(), "--profile", "local"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    workspace_dir = root / workspace_id
    config_path = workspace_dir / "config.yaml"
    (workspace_dir / "inputs" / "records.jsonl").write_text('{"id":"r1","sequence":"ACGT"}\n', encoding="utf-8")

    validate_result = _RUNNER.invoke(app, ["validate", "config", "--config", config_path.as_posix()])
    assert validate_result.exit_code == 0, validate_result.stdout

    dry_run_result = _RUNNER.invoke(app, ["run", "--config", config_path.as_posix(), "--dry-run"])
    assert dry_run_result.exit_code == 0, dry_run_result.stdout
    assert "Config validated (dry run)" in (dry_run_result.stdout or "")


def test_workspace_init_rejects_path_like_workspace_id(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"

    result = _RUNNER.invoke(app, ["workspace", "init", "--id", "bad/name", "--root", root.as_posix()])

    assert result.exit_code == 2
    assert "workspace id must be a simple directory name" in (result.stdout or "")


def test_workspace_init_fails_if_workspace_already_exists(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    workspace_dir = root / "demo_pressure"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    result = _RUNNER.invoke(app, ["workspace", "init", "--id", "demo_pressure", "--root", root.as_posix()])

    assert result.exit_code == 2
    assert "workspace already exists" in (result.stdout or "")
