"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_study_cli.py

CLI contract tests for the Study command group.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml
from typer.testing import CliRunner

import dnadesign.cruncher.cli.commands.study as study_cli
from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.study.manifest import (
    StudyManifestV1,
    StudyStatusV1,
    StudyTrialRun,
    write_study_manifest,
    write_study_status,
)

runner = CliRunner()


def test_root_help_includes_study_group() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "study" in result.output


def test_study_run_requires_spec_option() -> None:
    result = runner.invoke(app, ["study", "run"])
    assert result.exit_code != 0
    assert "--spec" in result.output


def test_study_run_resolves_relative_spec_from_init_cwd(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    workspace = repo_root / "workspaces" / "demo_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    expected_spec = workspace / "configs" / "studies" / "sweep.study.yaml"
    expected_spec.parent.mkdir(parents=True, exist_ok=True)
    expected_spec.write_text("study: {schema_version: 3}\n")
    emitted_run_dir = workspace / "outputs" / "studies" / "sweep" / "abc123"

    captured: dict[str, object] = {}

    def _fake_run_study(
        spec_path: Path,
        *,
        resume: bool = False,
        force_overwrite: bool = False,
        progress_bar: bool = True,
        quiet_logs: bool = False,
    ) -> Path:
        captured["spec_path"] = spec_path
        captured["resume"] = resume
        captured["force_overwrite"] = force_overwrite
        captured["progress_bar"] = progress_bar
        captured["quiet_logs"] = quiet_logs
        return emitted_run_dir

    monkeypatch.setattr(study_cli, "load_study_spec", lambda path: {"study": {"name": "sweep"}})
    monkeypatch.setattr(study_cli, "run_study", _fake_run_study)
    monkeypatch.setattr(
        study_cli,
        "study_show_payload",
        lambda path: {
            "study_name": "sweep",
            "study_id": "abc123",
            "status": "completed",
            "manifest_path": str(path / "study" / "study_manifest.json"),
            "status_path": str(path / "study" / "study_status.json"),
            "table_paths": [],
            "plot_paths": [],
            "success_runs": 1,
            "error_runs": 0,
            "pending_runs": 0,
        },
    )
    monkeypatch.chdir(repo_root)
    result = runner.invoke(
        app,
        ["study", "run", "--spec", "configs/studies/sweep.study.yaml"],
        env={"INIT_CWD": str(workspace)},
    )

    assert result.exit_code == 0
    assert captured["spec_path"] == expected_spec.resolve()
    assert captured["resume"] is False
    assert captured["force_overwrite"] is False
    assert captured["progress_bar"] is False
    assert captured["quiet_logs"] is False


def test_study_run_invalid_spec_fails_before_workflow_invocation(tmp_path: Path, monkeypatch) -> None:
    spec_path = tmp_path / "bad.study.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "study": {
                    "schema_version": 1,
                    "name": "bad_study",
                    "base_config": "config.yaml",
                    "target": {"kind": "regulator_set", "set_index": 1},
                    "replicates": {"seed_path": "sample.seed", "seeds": []},
                    "trials": [],
                }
            }
        )
    )
    called = {"run_study": False}

    def _fake_run_study(
        spec_path: Path,
        *,
        resume: bool = False,
        force_overwrite: bool = False,
        progress_bar: bool = True,
        quiet_logs: bool = False,
    ) -> Path:
        _ = (spec_path, resume, force_overwrite, progress_bar, quiet_logs)
        called["run_study"] = True
        raise AssertionError("run_study should not be invoked when study spec validation fails.")

    monkeypatch.setattr(study_cli, "run_study", _fake_run_study)
    result = runner.invoke(app, ["study", "run", "--spec", str(spec_path)])

    assert result.exit_code == 1
    assert "Study schema validation failed" in result.output
    assert "study.schema_version" in result.output
    assert called["run_study"] is False


def test_study_help_includes_clean_command() -> None:
    result = runner.invoke(app, ["study", "--help"])
    assert result.exit_code == 0
    assert "clean" in result.output
    assert "compact" in result.output


def test_study_list_help_includes_workspace_option() -> None:
    result = runner.invoke(app, ["study", "list", "--help"])
    assert result.exit_code == 0
    assert "--workspace" in result.output


def test_study_summarize_allow_partial_nonzero_policy_exits_nonzero(monkeypatch) -> None:
    run_dir = Path("/tmp/study")

    monkeypatch.setattr(
        study_cli,
        "summarize_study_run",
        lambda path, allow_partial: SimpleNamespace(
            study_run_dir=path,
            n_missing_total=2,
            n_missing_non_success=1,
            n_missing_run_dirs=1,
            n_missing_metric_artifacts=0,
            n_missing_mmr_tables=0,
            exit_code_policy="nonzero_if_any_error",
        ),
    )
    monkeypatch.setattr(
        study_cli,
        "study_show_payload",
        lambda path: {
            "study_name": "s",
            "study_id": "id",
            "status": "completed_with_errors",
            "manifest_path": str(path / "study" / "study_manifest.json"),
            "status_path": str(path / "study" / "study_status.json"),
            "table_paths": [],
            "plot_paths": [],
            "success_runs": 1,
            "error_runs": 1,
            "pending_runs": 0,
        },
    )

    result = runner.invoke(app, ["study", "summarize", "--run", str(run_dir), "--allow-partial"])
    assert result.exit_code == 1
    assert "partial data" in result.output


def test_study_summarize_allow_partial_always_zero_policy_exits_zero(monkeypatch) -> None:
    run_dir = Path("/tmp/study")

    monkeypatch.setattr(
        study_cli,
        "summarize_study_run",
        lambda path, allow_partial: SimpleNamespace(
            study_run_dir=path,
            n_missing_total=2,
            n_missing_non_success=1,
            n_missing_run_dirs=1,
            n_missing_metric_artifacts=0,
            n_missing_mmr_tables=0,
            exit_code_policy="always_zero",
        ),
    )
    monkeypatch.setattr(
        study_cli,
        "study_show_payload",
        lambda path: {
            "study_name": "s",
            "study_id": "id",
            "status": "completed_with_errors",
            "manifest_path": str(path / "study" / "study_manifest.json"),
            "status_path": str(path / "study" / "study_status.json"),
            "table_paths": [],
            "plot_paths": [],
            "success_runs": 1,
            "error_runs": 1,
            "pending_runs": 0,
        },
    )

    result = runner.invoke(app, ["study", "summarize", "--run", str(run_dir), "--allow-partial"])
    assert result.exit_code == 0


def _seed_workspace_with_study(tmp_path: Path) -> tuple[Path, Path]:
    workspace_root = tmp_path / "workspaces" / "demo_study_workspace"
    workspace_root.mkdir(parents=True)
    config_path = workspace_root / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "cruncher: {schema_version: 3, workspace: {out_dir: outputs, regulator_sets: [[lexA,cpxR]]}}\n"
    )

    spec_path = workspace_root / "configs" / "studies" / "sweep.study.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        yaml.safe_dump(
            {
                "study": {
                    "schema_version": 3,
                    "name": "sweep",
                    "base_config": "config.yaml",
                    "target": {"kind": "regulator_set", "set_index": 1},
                    "execution": {
                        "parallelism": 1,
                        "on_trial_error": "continue",
                        "exit_code_policy": "nonzero_if_any_error",
                        "summarize_after_run": True,
                    },
                    "artifacts": {"trial_output_profile": "minimal"},
                    "replicates": {"seed_path": "sample.seed", "seeds": [1]},
                    "trials": [{"id": "L6", "factors": {"sample.sequence_length": 6}}],
                    "replays": {"mmr_sweep": {"enabled": False}},
                }
            }
        )
    )

    run_dir = workspace_root / "outputs" / "studies" / "sweep" / "abc123"
    manifest_path = run_dir / "study" / "study_manifest.json"
    status_path = run_dir / "study" / "study_status.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_study_manifest(
        manifest_path,
        StudyManifestV1(
            study_name="sweep",
            study_id="abc123",
            spec_path=str(spec_path),
            spec_sha256="spec",
            base_config_path=str(config_path),
            base_config_sha256="config",
            created_at="2026-02-17T00:00:00+00:00",
            trial_runs=[
                StudyTrialRun(
                    trial_id="L6",
                    seed=1,
                    target_set_index=1,
                    target_tfs=["lexA", "cpxR"],
                    status="success",
                    run_dir=str(run_dir / "trials" / "L6" / "seed_1" / "run_a"),
                )
            ],
        ),
    )
    write_study_status(
        status_path,
        StudyStatusV1(
            study_name="sweep",
            study_id="abc123",
            status="completed",
            total_runs=1,
            pending_runs=0,
            running_runs=0,
            success_runs=1,
            error_runs=0,
            skipped_runs=0,
            warnings=[],
            started_at="2026-02-17T00:00:00+00:00",
            updated_at="2026-02-17T00:00:00+00:00",
            finished_at="2026-02-17T00:00:00+00:00",
        ),
    )

    return workspace_root, run_dir


def test_study_list_shows_workspace_scoped_specs_and_runs(tmp_path: Path) -> None:
    workspace_root, _ = _seed_workspace_with_study(tmp_path)
    root = workspace_root.parent
    result = runner.invoke(
        app,
        ["study", "list"],
        env={"CRUNCHER_WORKSPACE_ROOTS": str(root), "CRUNCHER_NONINTERACTIVE": "1", "COLUMNS": "220"},
    )
    assert result.exit_code == 0
    assert "Study Specs" in result.output
    assert "Study Runs" in result.output
    assert "specs live in <workspace>/configs/studies/*.study.yaml" in result.output
    assert "demo_study_workspace" in result.output
    assert "sweep" in result.output
    assert "outputs/studies" in result.output
    assert "abc123" in result.output
    assert "completed" in result.output


def test_study_list_accepts_workspace_path_without_roots(tmp_path: Path) -> None:
    workspace_root, _ = _seed_workspace_with_study(tmp_path)

    result = runner.invoke(
        app,
        ["study", "list", "--workspace", str(workspace_root)],
    )

    assert result.exit_code == 0
    assert "Study Specs" in result.output
    assert "Study Runs" in result.output
    assert "sweep" in result.output
    assert "abc123" in result.output


def test_study_clean_requires_id_or_all(tmp_path: Path) -> None:
    workspace_root, _ = _seed_workspace_with_study(tmp_path)
    root = workspace_root.parent
    result = runner.invoke(
        app,
        [
            "study",
            "clean",
            "--workspace",
            "demo_study_workspace",
            "--study",
            "sweep",
        ],
        env={"CRUNCHER_WORKSPACE_ROOTS": str(root), "CRUNCHER_NONINTERACTIVE": "1"},
    )
    assert result.exit_code == 1
    assert "Specify exactly one of --id or --all." in result.output


def test_study_clean_dry_run_keeps_outputs(tmp_path: Path) -> None:
    workspace_root, run_dir = _seed_workspace_with_study(tmp_path)
    root = workspace_root.parent
    spec_path = workspace_root / "configs" / "studies" / "sweep.study.yaml"
    assert run_dir.exists()
    assert spec_path.exists()

    result = runner.invoke(
        app,
        [
            "study",
            "clean",
            "--workspace",
            "demo_study_workspace",
            "--study",
            "sweep",
            "--id",
            "abc123",
        ],
        env={"CRUNCHER_WORKSPACE_ROOTS": str(root), "CRUNCHER_NONINTERACTIVE": "1"},
    )

    assert result.exit_code == 0
    assert "Dry run only." in result.output
    assert run_dir.exists()
    assert spec_path.exists()


def test_study_clean_confirm_deletes_outputs_keeps_spec(tmp_path: Path) -> None:
    workspace_root, run_dir = _seed_workspace_with_study(tmp_path)
    root = workspace_root.parent
    spec_path = workspace_root / "configs" / "studies" / "sweep.study.yaml"
    assert run_dir.exists()
    assert spec_path.exists()

    result = runner.invoke(
        app,
        [
            "study",
            "clean",
            "--workspace",
            "demo_study_workspace",
            "--study",
            "sweep",
            "--id",
            "abc123",
            "--confirm",
        ],
        env={"CRUNCHER_WORKSPACE_ROOTS": str(root), "CRUNCHER_NONINTERACTIVE": "1"},
    )

    assert result.exit_code == 0
    assert "Deleted 1 Study run directory." in result.output
    assert not run_dir.exists()
    assert spec_path.exists()


def test_study_clean_confirm_all_deletes_multiple_runs(tmp_path: Path) -> None:
    workspace_root, run_dir = _seed_workspace_with_study(tmp_path)
    root = workspace_root.parent

    second_run = workspace_root / "outputs" / "studies" / "sweep" / "def456"
    second_manifest = second_run / "study" / "study_manifest.json"
    second_status = second_run / "study" / "study_status.json"
    second_manifest.parent.mkdir(parents=True, exist_ok=True)
    second_manifest.write_text((run_dir / "study" / "study_manifest.json").read_text().replace("abc123", "def456"))
    second_status.write_text((run_dir / "study" / "study_status.json").read_text().replace("abc123", "def456"))
    assert run_dir.exists()
    assert second_run.exists()

    result = runner.invoke(
        app,
        [
            "study",
            "clean",
            "--workspace",
            "demo_study_workspace",
            "--study",
            "sweep",
            "--all",
            "--confirm",
        ],
        env={"CRUNCHER_WORKSPACE_ROOTS": str(root), "CRUNCHER_NONINTERACTIVE": "1"},
    )

    assert result.exit_code == 0
    assert "Deleted 2 Study run directories." in result.output
    assert not run_dir.exists()
    assert not second_run.exists()


def test_study_clean_accepts_workspace_path_without_roots(tmp_path: Path) -> None:
    workspace_root, run_dir = _seed_workspace_with_study(tmp_path)
    assert run_dir.exists()

    result = runner.invoke(
        app,
        [
            "study",
            "clean",
            "--workspace",
            str(workspace_root),
            "--study",
            "sweep",
            "--all",
        ],
    )

    assert result.exit_code == 0
    assert "Dry run only." in result.output
    assert run_dir.exists()


def test_study_clean_resolves_by_study_name_not_filename(tmp_path: Path) -> None:
    workspace_root, run_dir = _seed_workspace_with_study(tmp_path)
    original = workspace_root / "configs" / "studies" / "sweep.study.yaml"
    renamed = workspace_root / "configs" / "studies" / "arbitrary_name.study.yaml"
    original.rename(renamed)

    result = runner.invoke(
        app,
        [
            "study",
            "clean",
            "--workspace",
            str(workspace_root),
            "--study",
            "sweep",
            "--all",
        ],
    )

    assert result.exit_code == 0
    assert "Dry run only." in result.output
    assert run_dir.exists()


def test_study_compact_dry_run_uses_compaction_service(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "outputs" / "studies" / "sweep" / "abc123"
    run_dir.mkdir(parents=True, exist_ok=True)
    captured: dict[str, object] = {}

    def _fake_compact(path: Path, *, confirm: bool):
        captured["path"] = path
        captured["confirm"] = confirm
        return SimpleNamespace(
            trial_count=3,
            candidate_file_count=12,
            candidate_bytes=4096,
            deleted_file_count=0,
            deleted_bytes=0,
        )

    monkeypatch.setattr(study_cli, "compact_study_run", _fake_compact)
    result = runner.invoke(app, ["study", "compact", "--run", str(run_dir)])

    assert result.exit_code == 0
    assert captured["path"] == run_dir.resolve()
    assert captured["confirm"] is False
    assert "Study compact target" in result.output
    assert "Dry run only." in result.output
