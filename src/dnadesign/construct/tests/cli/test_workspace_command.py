"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/cli/test_workspace_command.py

Workspace command contracts for construct CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.construct.cli import app
from dnadesign.construct.src.seed import bootstrap_promoter_swap_demo
from dnadesign.construct.src.workspace import project_root

_RUNNER = CliRunner()


def test_workspace_where_uses_env_root_when_set(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    monkeypatch.setenv("CONSTRUCT_WORKSPACE_ROOT", root.as_posix())

    result = _RUNNER.invoke(app, ["workspace", "where"])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert f"workspace_root: {root.resolve()}" in output
    assert "workspace_root_source: env" in output
    assert "workspace_profile: blank" in output


def test_workspace_init_creates_default_layout_and_config(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    command_prefix = f"uv run --project {project_root()} construct"

    result = _RUNNER.invoke(app, ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix()])

    workspace_dir = root / "demo_construct"
    assert result.exit_code == 0, result.stdout
    assert (workspace_dir / "config.yaml").is_file()
    assert (workspace_dir / "construct.workspace.yaml").is_file()
    assert (workspace_dir / "inputs" / "README.md").is_file()
    assert (workspace_dir / "inputs" / "import_manifest.template.yaml").is_file()
    assert (workspace_dir / "outputs" / "logs" / "ops" / "audit").is_dir()
    assert "# root: outputs/usr_datasets" in (workspace_dir / "config.yaml").read_text(encoding="utf-8")
    inputs_readme = (workspace_dir / "inputs" / "README.md").read_text(encoding="utf-8")
    assert "workspace's `outputs/usr_datasets/` root" in inputs_readme
    assert "src/dnadesign/usr/datasets/" in inputs_readme
    output = result.stdout or ""
    assert "profile: blank" in output
    assert "workspace_registry:" in output
    assert f"{command_prefix} workspace show --workspace" in output
    assert f"{command_prefix} validate config --config" in output
    assert "import_manifest.template.yaml" in output
    assert f"Optional demo bootstrap: {command_prefix} seed promoter-swap-demo" in output
    assert "outputs/usr_datasets" in output
    assert "./runbook.sh" not in output


def test_workspace_init_copies_packaged_promoter_swap_demo_profile(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    command_prefix = f"uv run --project {project_root()} construct"

    result = _RUNNER.invoke(
        app,
        ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix(), "--profile", "promoter-swap-demo"],
    )

    workspace_dir = root / "demo_construct"
    assert result.exit_code == 0, result.stdout
    assert (workspace_dir / "README.md").is_file()
    assert (workspace_dir / "construct.workspace.yaml").is_file()
    assert (workspace_dir / "runbook.md").is_file()
    assert (workspace_dir / "runbook.sh").is_file()
    assert (workspace_dir / "config.slot_a.window.yaml").is_file()
    assert (workspace_dir / "config.slot_b.full.yaml").is_file()
    assert (workspace_dir / "inputs" / "README.md").is_file()
    assert (workspace_dir / "outputs" / "logs" / "ops" / "audit").is_dir()
    assert "root: outputs/usr_datasets" in (workspace_dir / "config.slot_a.window.yaml").read_text(encoding="utf-8")
    assert "workspace-local USR root `outputs/usr_datasets/`" in (workspace_dir / "inputs" / "README.md").read_text(
        encoding="utf-8"
    )
    output = result.stdout or ""
    assert "profile: promoter-swap-demo" in output
    assert "workspace_registry:" in output
    assert f"{command_prefix} workspace show --workspace" in output
    assert "choose one of the packaged config.*.yaml files" in output
    assert "--root" in output
    assert "outputs/usr_datasets" in output
    assert "./runbook.sh --mode dry-run --config <chosen-config>" in output


def test_workspace_init_rejects_path_like_workspace_id(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"

    result = _RUNNER.invoke(app, ["workspace", "init", "--id", "bad/name", "--root", root.as_posix()])

    assert result.exit_code == 2
    assert "workspace id must be a simple directory name" in (result.stdout or "")


def test_workspace_init_fails_if_workspace_already_exists(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    workspace_dir = root / "demo_construct"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    result = _RUNNER.invoke(app, ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix()])

    assert result.exit_code == 2
    assert "workspace already exists" in (result.stdout or "")


def test_workspace_show_reports_registry_summary(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"

    init_result = _RUNNER.invoke(
        app,
        ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix(), "--profile", "promoter-swap-demo"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    workspace_dir = root / "demo_construct"
    result = _RUNNER.invoke(app, ["workspace", "show", "--workspace", workspace_dir.as_posix()])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert "workspace_id: demo_construct" in output
    assert "profile: promoter-swap-demo" in output
    assert "shared_usr_root: src/dnadesign/usr/datasets (repo-relative hint)" in output
    assert "workspace_usr_root: outputs/usr_datasets (workspace-relative default)" in output
    assert "project: id=slot_a_window" in output


def test_workspace_doctor_reports_ok_for_packaged_demo(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    init_result = _RUNNER.invoke(
        app,
        ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix(), "--profile", "promoter-swap-demo"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    workspace_dir = root / "demo_construct"
    result = _RUNNER.invoke(app, ["workspace", "doctor", "--workspace", workspace_dir.as_posix()])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert "projects_checked: 4" in output
    assert "issues_total: 0" in output
    assert "workspace_doctor: ok" in output


def test_workspace_doctor_reports_registry_drift(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    init_result = _RUNNER.invoke(
        app,
        ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix(), "--profile", "promoter-swap-demo"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    workspace_dir = root / "demo_construct"
    registry_path = workspace_dir / "construct.workspace.yaml"
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    payload["workspace"]["projects"][0]["output_dataset"] = "drifted_output_dataset"
    registry_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = _RUNNER.invoke(app, ["workspace", "doctor", "--workspace", workspace_dir.as_posix()])

    assert result.exit_code == 1, result.stdout
    assert "does not match config output.dataset" in (result.stdout or "")


def test_workspace_doctor_rejects_project_config_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    init_result = _RUNNER.invoke(
        app,
        ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix(), "--profile", "promoter-swap-demo"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    workspace_dir = root / "demo_construct"
    registry_path = workspace_dir / "construct.workspace.yaml"
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    payload["workspace"]["projects"][0]["config"] = "../escaped.yaml"
    registry_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = _RUNNER.invoke(app, ["workspace", "doctor", "--workspace", workspace_dir.as_posix()])

    assert result.exit_code == 2, result.stdout
    assert "must stay inside the workspace root" in (result.stdout or "")


def test_workspace_validate_project_runtime_resolves_registry_project(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    init_result = _RUNNER.invoke(
        app,
        ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix(), "--profile", "promoter-swap-demo"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    workspace_dir = root / "demo_construct"
    bootstrap_promoter_swap_demo(
        root=workspace_dir / "outputs" / "usr_datasets",
        manifest=workspace_dir / "inputs" / "seed_manifest.yaml",
    )

    result = _RUNNER.invoke(
        app,
        [
            "workspace",
            "validate-project",
            "--workspace",
            workspace_dir.as_posix(),
            "--project",
            "slot_a_window",
            "--runtime",
        ],
    )

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert "job_id: promoter_swap_slot_a_window_1kb" in output
    assert "template_id: pDual-10" in output
    assert "placement: part=anchor" in output
    assert "rows_total: 4" in output


def test_workspace_run_project_dry_run_resolves_registry_project(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    init_result = _RUNNER.invoke(
        app,
        ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix(), "--profile", "promoter-swap-demo"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    workspace_dir = root / "demo_construct"
    bootstrap_promoter_swap_demo(
        root=workspace_dir / "outputs" / "usr_datasets",
        manifest=workspace_dir / "inputs" / "seed_manifest.yaml",
    )

    result = _RUNNER.invoke(
        app,
        [
            "workspace",
            "run-project",
            "--workspace",
            workspace_dir.as_posix(),
            "--project",
            "slot_a_window",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert "Config validated (dry run): job=promoter_swap_slot_a_window_1kb" in output
    assert "output_dataset: pdual10_slot_a_window_1kb_demo" in output
