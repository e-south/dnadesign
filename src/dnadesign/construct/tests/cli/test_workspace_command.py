"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/cli/test_workspace_command.py

Workspace command contracts for construct CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.construct.cli import app
from dnadesign.construct.src.seed import bootstrap_anchor_template_demo
from dnadesign.construct.src.workspace import project_root
from dnadesign.usr import Dataset

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


def test_workspace_where_defaults_to_cwd_when_unset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CONSTRUCT_WORKSPACE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    result = _RUNNER.invoke(app, ["workspace", "where"])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert f"workspace_root: {tmp_path.resolve()}" in output
    assert "workspace_root_source: cwd" in output


def test_workspace_init_creates_default_layout_and_config(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    command_prefix = f"uv run --project {shlex.quote(project_root().as_posix())} construct"

    result = _RUNNER.invoke(app, ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix()])

    workspace_dir = root / "demo_construct"
    assert result.exit_code == 0, result.stdout
    assert (workspace_dir / "config.yaml").is_file()
    assert (workspace_dir / "construct.workspace.yaml").is_file()
    assert (workspace_dir / "inputs" / "README.md").is_file()
    assert (workspace_dir / "inputs" / "import_manifest.template.yaml").is_file()
    assert (workspace_dir / "outputs" / "logs" / "ops" / "audit").is_dir()
    assert "root: outputs/usr_datasets" in (workspace_dir / "config.yaml").read_text(encoding="utf-8")
    inputs_readme = (workspace_dir / "inputs" / "README.md").read_text(encoding="utf-8")
    assert "workspace's `outputs/usr_datasets/` root" in inputs_readme
    assert "src/dnadesign/usr/datasets/" in inputs_readme
    output = result.stdout or ""
    assert "profile: blank" in output
    assert "workspace_registry:" in output
    assert f"{command_prefix} workspace show --workspace" in output
    assert f"{command_prefix} validate config --config" in output
    assert "import_manifest.template.yaml" in output
    assert f"Then: {command_prefix} seed import-manifest" in output
    assert "outputs/usr_datasets" in output
    assert "seed_manifest.yaml" not in output
    assert "--profile anchor-template-demo" in output
    assert "./runbook.sh" not in output


def test_workspace_init_without_root_uses_cwd(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CONSTRUCT_WORKSPACE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    result = _RUNNER.invoke(app, ["workspace", "init", "--id", "demo_construct"])

    workspace_dir = tmp_path / "demo_construct"
    assert result.exit_code == 0, result.stdout
    assert workspace_dir.is_dir()
    assert (workspace_dir / "construct.workspace.yaml").is_file()


def test_workspace_init_copies_packaged_promoter_swap_demo_profile(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    command_prefix = f"uv run --project {shlex.quote(project_root().as_posix())} construct"

    result = _RUNNER.invoke(
        app,
        ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix(), "--profile", "anchor-template-demo"],
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
    assert "profile: anchor-template-demo" in output
    assert "workspace_registry:" in output
    assert f"{command_prefix} workspace show --workspace" in output
    assert "choose one of the packaged config.*.yaml files" in output
    assert "--root" in output
    assert "outputs/usr_datasets" in output
    assert "./runbook.sh --mode dry-run --config <chosen-config>" in output


def test_workspace_init_copies_packaged_source_of_truth_demo_profile(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    command_prefix = f"uv run --project {shlex.quote(project_root().as_posix())} construct"

    result = _RUNNER.invoke(
        app,
        [
            "workspace",
            "init",
            "--id",
            "demo_construct",
            "--root",
            root.as_posix(),
            "--profile",
            "anchor-template-shared-dataset-demo",
        ],
    )

    workspace_dir = root / "demo_construct"
    assert result.exit_code == 0, result.stdout
    assert (workspace_dir / "README.md").is_file()
    assert (workspace_dir / "construct.workspace.yaml").is_file()
    assert (workspace_dir / "runbook.md").is_file()
    assert (workspace_dir / "runbook.sh").is_file()
    assert (workspace_dir / "config.slot_a.window.yaml").is_file()
    assert (workspace_dir / "config.slot_b.window.yaml").is_file()
    assert not (workspace_dir / "config.slot_a.full.yaml").exists()
    assert not (workspace_dir / "config.slot_b.full.yaml").exists()
    output = result.stdout or ""
    assert "profile: anchor-template-shared-dataset-demo" in output
    assert "workspace_registry:" in output
    assert f"{command_prefix} workspace show --workspace" in output
    assert "outputs/usr_datasets" in output
    assert "./runbook.sh --mode dry-run-all" in output


def test_workspace_init_quotes_project_root_in_external_workspace_commands(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "ws_root"
    fake_repo_root = tmp_path / "repo with spaces"
    expected_prefix = f"uv run --project {shlex.quote(fake_repo_root.as_posix())} construct"
    monkeypatch.setattr("dnadesign.construct.src.cli.commands.workspace.project_root_or_none", lambda: fake_repo_root)

    result = _RUNNER.invoke(app, ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix()])

    assert result.exit_code == 0, result.stdout
    assert expected_prefix in (result.stdout or "")


def test_workspace_init_uses_plain_uv_run_when_repo_checkout_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "ws_root"
    monkeypatch.setattr("dnadesign.construct.src.cli.commands.workspace.project_root_or_none", lambda: None)
    monkeypatch.setattr("dnadesign.construct.src.workspace.project_root_or_none", lambda: None)

    result = _RUNNER.invoke(
        app,
        ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix(), "--profile", "anchor-template-demo"],
    )

    workspace_dir = root / "demo_construct"
    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert "uv run construct workspace show --workspace" in output
    assert "uv run --project" not in output
    runbook = (workspace_dir / "runbook.sh").read_text(encoding="utf-8")
    assert "__CONSTRUCT_PROJECT_ROOT__" not in runbook
    assert 'PROJECT_ROOT="${CONSTRUCT_RUNBOOK_PROJECT_ROOT:-}"' in runbook


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


def test_workspace_init_rejects_file_root(tmp_path: Path) -> None:
    root_file = tmp_path / "workspace_root.txt"
    root_file.write_text("not a directory\n", encoding="utf-8")

    result = _RUNNER.invoke(app, ["workspace", "init", "--id", "demo_construct", "--root", root_file.as_posix()])

    assert result.exit_code == 2
    assert "workspace root must be a directory" in (result.stdout or "")


def test_workspace_init_cleans_up_partial_workspace_on_copy_failure(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "ws_root"

    def _boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("simulated scaffold failure")

    monkeypatch.setattr("dnadesign.construct.src.workspace._copy_blank_workspace", _boom)

    result = _RUNNER.invoke(app, ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix()])

    workspace_dir = root / "demo_construct"
    assert result.exit_code == 2
    assert "construct workspace could not be created" in (result.stdout or "")
    assert not workspace_dir.exists()


def test_workspace_show_reports_registry_summary(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"

    init_result = _RUNNER.invoke(
        app,
        ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix(), "--profile", "anchor-template-demo"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    workspace_dir = root / "demo_construct"
    result = _RUNNER.invoke(app, ["workspace", "show", "--workspace", workspace_dir.as_posix()])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert "workspace_id: demo_construct" in output
    assert "profile: anchor-template-demo" in output
    assert "shared_usr_root: src/dnadesign/usr/datasets (repo-relative hint)" in output
    assert "workspace_usr_root: outputs/usr_datasets (workspace-relative default)" in output
    assert "project: id=slot_a_window" in output


def test_workspace_list_json_reports_packaged_workspace_state(monkeypatch, tmp_path: Path) -> None:
    construct_root = tmp_path / "construct_root"
    workspaces_root = construct_root / "workspaces"
    for workspace_id in ("demo_anchor_template_local", "demo_anchor_template_shared_dataset"):
        workspace_dir = workspaces_root / workspace_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "construct.workspace.yaml").write_text(
            "workspace:\n  id: demo\n  profile: demo\n  projects: []\n",
            encoding="utf-8",
        )
    (workspaces_root / "demo_anchor_template_shared_dataset" / "outputs" / "logs").mkdir(
        parents=True,
        exist_ok=True,
    )
    (workspaces_root / "demo_anchor_template_shared_dataset" / "outputs" / "logs" / "run.log").write_text(
        "ok\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("dnadesign.construct.src.workspace._construct_root", lambda: construct_root)

    result = _RUNNER.invoke(app, ["workspace", "list", "--format", "json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    by_id = {entry["workspace_id"]: entry for entry in payload}
    assert by_id["demo_anchor_template_local"]["workspace_state"] == "clean"
    assert by_id["demo_anchor_template_local"]["output_files"] == 0
    assert by_id["demo_anchor_template_local"]["workspace_source"] == "packaged"
    assert by_id["demo_anchor_template_shared_dataset"]["workspace_state"] == "attention"
    assert by_id["demo_anchor_template_shared_dataset"]["output_files"] == 1
    assert by_id["demo_anchor_template_shared_dataset"]["workspace_source"] == "packaged"
    assert by_id["demo_anchor_template_shared_dataset"]["latest_output_mtime"] is not None


def test_workspace_list_json_reports_local_copied_workspace_state(monkeypatch, tmp_path: Path) -> None:
    construct_root = tmp_path / "construct_root"
    packaged_root = construct_root / "workspaces" / "demo_anchor_template_local"
    packaged_root.mkdir(parents=True, exist_ok=True)
    (packaged_root / "construct.workspace.yaml").write_text(
        "workspace:\n  id: demo_anchor_template_local\n  profile: demo\n  projects: []\n",
        encoding="utf-8",
    )
    local_workspace = tmp_path / "demo_construct"
    (local_workspace / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
    (local_workspace / "construct.workspace.yaml").write_text(
        "workspace:\n  id: demo_construct\n  profile: anchor-template-demo\n  projects: []\n",
        encoding="utf-8",
    )
    (local_workspace / "outputs" / "logs" / "run.log").write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr("dnadesign.construct.src.workspace._construct_root", lambda: construct_root)
    monkeypatch.chdir(tmp_path)

    result = _RUNNER.invoke(app, ["workspace", "list", "--format", "json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    by_id = {entry["workspace_id"]: entry for entry in payload}
    assert by_id["demo_construct"]["workspace_source"] == "local"
    assert by_id["demo_construct"]["workspace_state"] == "attention"
    assert by_id["demo_construct"]["output_files"] == 1
    assert by_id["demo_construct"]["workspace_root_source"] == "cwd"
    assert by_id["demo_construct"]["workspace_dir"] == str(local_workspace.resolve())
    assert by_id["demo_anchor_template_local"]["workspace_source"] == "packaged"


def test_workspace_doctor_reports_ok_for_packaged_demo(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    init_result = _RUNNER.invoke(
        app,
        ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix(), "--profile", "anchor-template-demo"],
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
        ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix(), "--profile", "anchor-template-demo"],
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
        ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix(), "--profile", "anchor-template-demo"],
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
        ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix(), "--profile", "anchor-template-demo"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    workspace_dir = root / "demo_construct"
    bootstrap_anchor_template_demo(
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
    assert "job_id: anchor_template_slot_a_window_1kb" in output
    assert "template_id: template_backbone_dual_slot" in output
    assert "placement: part=anchor" in output
    assert "rows_total: 4" in output


def test_workspace_show_resolves_local_workspace_id_from_active_root(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    init_result = _RUNNER.invoke(
        app,
        ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix(), "--profile", "anchor-template-demo"],
    )
    assert init_result.exit_code == 0, init_result.stdout
    monkeypatch.setenv("CONSTRUCT_WORKSPACE_ROOT", root.as_posix())

    result = _RUNNER.invoke(app, ["workspace", "show", "--workspace", "demo_construct"])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert f"workspace_registry: {root / 'demo_construct' / 'construct.workspace.yaml'}" in output
    assert "workspace_id: demo_construct" in output


def test_workspace_show_resolves_packaged_workspace_id(monkeypatch, tmp_path: Path) -> None:
    construct_root = tmp_path / "construct_root"
    packaged_workspace = construct_root / "workspaces" / "demo_anchor_template_local"
    packaged_workspace.mkdir(parents=True, exist_ok=True)
    (packaged_workspace / "construct.workspace.yaml").write_text(
        "workspace:\n  id: demo_anchor_template_local\n  profile: anchor-template-demo\n  projects: []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("dnadesign.construct.src.workspace._construct_root", lambda: construct_root)

    result = _RUNNER.invoke(app, ["workspace", "show", "--workspace", "demo_anchor_template_local"])

    assert result.exit_code == 0, result.stdout
    output = result.stdout or ""
    assert f"workspace_registry: {packaged_workspace / 'construct.workspace.yaml'}" in output
    assert "workspace_id: demo_anchor_template_local" in output


def test_workspace_run_project_dry_run_resolves_registry_project(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    init_result = _RUNNER.invoke(
        app,
        ["workspace", "init", "--id", "demo_construct", "--root", root.as_posix(), "--profile", "anchor-template-demo"],
    )
    assert init_result.exit_code == 0, init_result.stdout

    workspace_dir = root / "demo_construct"
    bootstrap_anchor_template_demo(
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
    assert "Config validated (dry run): job=anchor_template_slot_a_window_1kb" in output
    assert "output_dataset: anchor_template_slot_a_window_1kb_demo" in output


def test_workspace_shared_source_of_truth_profile_accumulates_distinct_projects(tmp_path: Path) -> None:
    root = tmp_path / "ws_root"
    init_result = _RUNNER.invoke(
        app,
        [
            "workspace",
            "init",
            "--id",
            "demo_construct",
            "--root",
            root.as_posix(),
            "--profile",
            "anchor-template-shared-dataset-demo",
        ],
    )
    assert init_result.exit_code == 0, init_result.stdout

    workspace_dir = root / "demo_construct"
    bootstrap_anchor_template_demo(
        root=workspace_dir / "outputs" / "usr_datasets",
        manifest=workspace_dir / "inputs" / "seed_manifest.yaml",
    )

    for project in ("slot_a_window", "slot_b_window"):
        validate_result = _RUNNER.invoke(
            app,
            [
                "workspace",
                "validate-project",
                "--workspace",
                workspace_dir.as_posix(),
                "--project",
                project,
                "--runtime",
            ],
        )
        assert validate_result.exit_code == 0, validate_result.stdout

        run_result = _RUNNER.invoke(
            app,
            [
                "workspace",
                "run-project",
                "--workspace",
                workspace_dir.as_posix(),
                "--project",
                project,
            ],
        )
        assert run_result.exit_code == 0, run_result.stdout

    output_ds = Dataset(workspace_dir / "outputs" / "usr_datasets", "anchor_template_shared_dataset_demo")
    output_ds.validate(strict=True)
    frame = output_ds.head(n=20)
    assert len(frame) == 8
    assert set(frame["construct__job"]) == {"anchor_template_slot_a_window_1kb", "anchor_template_slot_b_window_1kb"}
    assert "usr_label__primary" in frame.columns
