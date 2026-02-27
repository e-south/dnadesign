"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_sync_target_modes.py

Tests for sync target mode classification and dataset directory resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dnadesign.usr.src.cli_commands import sync as sync_commands


def test_is_file_mode_target_treats_namespaced_dataset_id_as_dataset() -> None:
    assert sync_commands._is_file_mode_target("densegen/demo") is False
    assert sync_commands._is_file_mode_target("demo") is False
    assert sync_commands._is_file_mode_target("records.parquet") is True


def test_resolve_dataset_dir_prefers_root_when_relative(tmp_path: Path) -> None:
    root = tmp_path / "usr_root"
    dataset_dir = root / "densegen" / "demo"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "records.parquet").write_text("stub", encoding="utf-8")
    resolved_root, dataset_id = sync_commands._resolve_dataset_dir_target(dataset_dir, root)
    assert resolved_root == root
    assert dataset_id == "densegen/demo"


def test_resolve_dataset_dir_uses_registry_ancestor(tmp_path: Path) -> None:
    root = tmp_path / "workspace" / "outputs" / "usr_datasets"
    root.mkdir(parents=True)
    (root / "registry.yaml").write_text("namespaces: {}\n", encoding="utf-8")
    dataset_dir = root / "densegen" / "demo_hpc"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "records.parquet").write_text("stub", encoding="utf-8")

    resolved_root, dataset_id = sync_commands._resolve_dataset_dir_target(dataset_dir, tmp_path / "other_root")
    assert resolved_root == root
    assert dataset_id == "densegen/demo_hpc"


def test_cmd_diff_accepts_dataset_directory_path(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "outputs" / "usr_datasets"
    root.mkdir(parents=True)
    (root / "registry.yaml").write_text("namespaces: {}\n", encoding="utf-8")
    dataset_dir = root / "densegen" / "demo_hpc"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "records.parquet").write_text("stub", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_plan_diff(root: Path, dataset: str, remote_name: str, *, verify: str):
        captured["root"] = root
        captured["dataset"] = dataset
        captured["remote"] = remote_name
        captured["verify"] = verify
        return SimpleNamespace(has_change=False, verify_notes=[])

    monkeypatch.setattr(sync_commands, "plan_diff", _fake_plan_diff)
    monkeypatch.setattr(sync_commands, "_print_verify_notes", lambda _summary: None)
    monkeypatch.setattr(sync_commands, "_print_diff", lambda _summary, *, use_rich=None: None)

    args = SimpleNamespace(
        dataset=str(dataset_dir),
        remote="cluster",
        verify="size",
        root=tmp_path / "ignored",
        rich=False,
        repo_root=None,
        remote_path=None,
    )
    sync_commands.cmd_diff(
        args,
        resolve_output_format=lambda _args: "plain",
        print_json=lambda _payload: None,
        output_version=1,
    )

    assert captured["root"] == root
    assert captured["dataset"] == "densegen/demo_hpc"
    assert captured["remote"] == "cluster"
    assert captured["verify"] == "size"


def test_cmd_pull_accepts_densegen_workspace_dataset_directory_path(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "workspace" / "outputs" / "usr_datasets"
    root.mkdir(parents=True)
    (root / "registry.yaml").write_text("namespaces: {}\n", encoding="utf-8")
    dataset_dir = root / "densegen" / "demo_hpc"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "records.parquet").write_text("stub", encoding="utf-8")
    (dataset_dir / "_derived" / "densegen").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "_artifacts" / "pending_overlay").mkdir(parents=True, exist_ok=True)

    summary = SimpleNamespace(has_change=False, verify_notes=[])
    captured: dict[str, object] = {}

    monkeypatch.setattr(sync_commands, "plan_diff", lambda *_args, **_kwargs: summary)
    monkeypatch.setattr(sync_commands, "_print_verify_notes", lambda _summary: None)
    monkeypatch.setattr(sync_commands, "_print_diff", lambda _summary, *, use_rich=None: None)
    monkeypatch.setattr(sync_commands, "_confirm_or_abort", lambda _summary, *, assume_yes: None)

    def _fake_execute_pull(root: Path, dataset: str, remote_name: str, opts):
        captured["root"] = root
        captured["dataset"] = dataset
        captured["remote"] = remote_name
        captured["opts"] = opts
        return summary

    monkeypatch.setattr(sync_commands, "execute_pull", _fake_execute_pull)

    args = SimpleNamespace(
        dataset=str(dataset_dir),
        remote="bu-scc",
        verify="auto",
        root=tmp_path / "ignored",
        rich=False,
        repo_root=None,
        remote_path=None,
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
    )

    sync_commands.cmd_pull(args)

    assert captured["root"] == root
    assert captured["dataset"] == "densegen/demo_hpc"
    assert captured["remote"] == "bu-scc"
    assert getattr(captured["opts"], "verify", None) == "auto"


def test_cmd_diff_accepts_namespaced_dataset_id_when_local_dataset_missing(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_plan_diff(root: Path, dataset: str, remote_name: str, *, verify: str):
        captured["root"] = root
        captured["dataset"] = dataset
        captured["remote"] = remote_name
        captured["verify"] = verify
        return SimpleNamespace(has_change=False, verify_notes=[])

    monkeypatch.setattr(sync_commands, "plan_diff", _fake_plan_diff)
    monkeypatch.setattr(sync_commands, "_print_verify_notes", lambda _summary: None)
    monkeypatch.setattr(sync_commands, "_print_diff", lambda _summary, *, use_rich=None: None)

    args = SimpleNamespace(
        dataset="densegen/demo_hpc_remote_only",
        remote="cluster",
        verify="auto",
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path=None,
    )
    sync_commands.cmd_diff(
        args,
        resolve_output_format=lambda _args: "plain",
        print_json=lambda _payload: None,
        output_version=1,
    )

    assert captured["root"] == args.root
    assert captured["dataset"] == "densegen/demo_hpc_remote_only"
    assert captured["remote"] == "cluster"
    assert captured["verify"] == "auto"


def test_cmd_pull_accepts_namespaced_dataset_id_when_local_dataset_missing(tmp_path: Path, monkeypatch) -> None:
    summary = SimpleNamespace(has_change=False, verify_notes=[])
    captured: dict[str, object] = {}

    monkeypatch.setattr(sync_commands, "plan_diff", lambda *_args, **_kwargs: summary)
    monkeypatch.setattr(sync_commands, "_print_verify_notes", lambda _summary: None)
    monkeypatch.setattr(sync_commands, "_print_diff", lambda _summary, *, use_rich=None: None)
    monkeypatch.setattr(sync_commands, "_confirm_or_abort", lambda _summary, *, assume_yes: None)

    def _fake_execute_pull(root: Path, dataset: str, remote_name: str, opts):
        captured["root"] = root
        captured["dataset"] = dataset
        captured["remote"] = remote_name
        captured["opts"] = opts
        return summary

    monkeypatch.setattr(sync_commands, "execute_pull", _fake_execute_pull)

    args = SimpleNamespace(
        dataset="densegen/demo_hpc_remote_only",
        remote="bu-scc",
        verify="auto",
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path=None,
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
        verify_sidecars=False,
    )

    sync_commands.cmd_pull(args)

    assert captured["root"] == args.root
    assert captured["dataset"] == "densegen/demo_hpc_remote_only"
    assert captured["remote"] == "bu-scc"
    assert getattr(captured["opts"], "verify", None) == "auto"
    assert getattr(captured["opts"], "verify_sidecars", None) is False


def test_cmd_pull_passes_verify_sidecars_option(tmp_path: Path, monkeypatch) -> None:
    summary = SimpleNamespace(has_change=False, verify_notes=[])
    captured: dict[str, object] = {}

    monkeypatch.setattr(sync_commands, "plan_diff", lambda *_args, **_kwargs: summary)
    monkeypatch.setattr(sync_commands, "_print_verify_notes", lambda _summary: None)
    monkeypatch.setattr(sync_commands, "_print_diff", lambda _summary, *, use_rich=None: None)
    monkeypatch.setattr(sync_commands, "_confirm_or_abort", lambda _summary, *, assume_yes: None)

    def _fake_execute_pull(root: Path, dataset: str, remote_name: str, opts):
        captured["root"] = root
        captured["dataset"] = dataset
        captured["remote"] = remote_name
        captured["opts"] = opts
        return summary

    monkeypatch.setattr(sync_commands, "execute_pull", _fake_execute_pull)

    args = SimpleNamespace(
        dataset="densegen/demo_hpc_remote_only",
        remote="bu-scc",
        verify="auto",
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path=None,
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
        verify_sidecars=True,
    )

    sync_commands.cmd_pull(args)

    assert captured["root"] == args.root
    assert captured["dataset"] == "densegen/demo_hpc_remote_only"
    assert captured["remote"] == "bu-scc"
    assert getattr(captured["opts"], "verify", None) == "auto"
    assert getattr(captured["opts"], "verify_sidecars", None) is True


def test_cmd_pull_sync_audit_uses_post_execution_summary(tmp_path: Path, monkeypatch) -> None:
    pre_summary = SimpleNamespace(has_change=True, verify_notes=[])
    post_summary = SimpleNamespace(has_change=False, verify_notes=[])
    captured: dict[str, object] = {}

    monkeypatch.setattr(sync_commands, "plan_diff", lambda *_args, **_kwargs: pre_summary)
    monkeypatch.setattr(sync_commands, "_print_verify_notes", lambda _summary: None)
    monkeypatch.setattr(sync_commands, "_print_diff", lambda _summary, *, use_rich=None: None)
    monkeypatch.setattr(sync_commands, "_confirm_or_abort", lambda _summary, *, assume_yes: None)
    monkeypatch.setattr(sync_commands, "execute_pull", lambda *_args, **_kwargs: post_summary)
    monkeypatch.setattr(
        sync_commands,
        "_print_sync_audit",
        lambda summary, *, action, dry_run, verify_sidecars: captured.update({"summary": summary, "action": action}),
    )

    args = SimpleNamespace(
        dataset="densegen/demo_hpc_remote_only",
        remote="bu-scc",
        verify="auto",
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path=None,
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
        verify_sidecars=False,
    )

    sync_commands.cmd_pull(args)

    assert captured["action"] == "pull"
    assert captured["summary"] is post_summary


def test_cmd_push_sync_audit_uses_post_execution_summary(tmp_path: Path, monkeypatch) -> None:
    pre_summary = SimpleNamespace(has_change=True, verify_notes=[])
    post_summary = SimpleNamespace(has_change=False, verify_notes=[])
    captured: dict[str, object] = {}

    monkeypatch.setattr(sync_commands, "resolve_dataset_name_interactive", lambda *_args, **_kwargs: "densegen/demo")
    monkeypatch.setattr(sync_commands, "plan_diff", lambda *_args, **_kwargs: pre_summary)
    monkeypatch.setattr(sync_commands, "_print_verify_notes", lambda _summary: None)
    monkeypatch.setattr(sync_commands, "_print_diff", lambda _summary, *, use_rich=None: None)
    monkeypatch.setattr(sync_commands, "_confirm_or_abort", lambda _summary, *, assume_yes: None)
    monkeypatch.setattr(sync_commands, "execute_push", lambda *_args, **_kwargs: post_summary)
    monkeypatch.setattr(
        sync_commands,
        "_print_sync_audit",
        lambda summary, *, action, dry_run, verify_sidecars: captured.update({"summary": summary, "action": action}),
    )

    args = SimpleNamespace(
        dataset="densegen/demo",
        remote="bu-scc",
        verify="auto",
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path=None,
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
        verify_sidecars=False,
    )

    sync_commands.cmd_push(args)

    assert captured["action"] == "push"
    assert captured["summary"] is post_summary


def test_cmd_pull_file_mode_rejects_verify_sidecars(tmp_path: Path) -> None:
    file_target = tmp_path / "records.parquet"
    file_target.write_text("stub", encoding="utf-8")
    args = SimpleNamespace(
        dataset=str(file_target),
        remote="bu-scc",
        verify="auto",
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path="/remote/path/records.parquet",
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
        verify_sidecars=True,
    )

    try:
        sync_commands.cmd_pull(args)
    except SystemExit as exc:
        assert "dataset-only flags" in str(exc)
        return
    raise AssertionError("expected verify-sidecars in FILE mode to fail fast")


def test_cmd_pull_strict_bootstrap_requires_namespaced_dataset_id(tmp_path: Path) -> None:
    args = SimpleNamespace(
        dataset="demo_remote_only",
        remote="bu-scc",
        verify="auto",
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path=None,
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
        strict_bootstrap_id=True,
    )

    try:
        sync_commands.cmd_pull(args)
    except SystemExit as exc:
        assert "namespace-qualified dataset id" in str(exc)
        return
    raise AssertionError("expected strict bootstrap pull to require namespace-qualified id")


def test_cmd_pull_strict_bootstrap_env_requires_namespaced_dataset_id(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("USR_SYNC_STRICT_BOOTSTRAP_ID", "1")
    args = SimpleNamespace(
        dataset="demo_remote_only",
        remote="bu-scc",
        verify="auto",
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path=None,
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
        strict_bootstrap_id=False,
    )

    try:
        sync_commands.cmd_pull(args)
    except SystemExit as exc:
        assert "namespace-qualified dataset id" in str(exc)
        return
    raise AssertionError("expected env strict bootstrap pull to require namespace-qualified id")


def test_cmd_pull_strict_bootstrap_allows_unqualified_existing_local_dataset(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "usr_root"
    dataset_dir = root / "demo"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "records.parquet").write_text("stub", encoding="utf-8")

    summary = SimpleNamespace(has_change=False, verify_notes=[])
    captured: dict[str, object] = {}

    monkeypatch.setattr(sync_commands, "plan_diff", lambda *_args, **_kwargs: summary)
    monkeypatch.setattr(sync_commands, "_print_verify_notes", lambda _summary: None)
    monkeypatch.setattr(sync_commands, "_print_diff", lambda _summary, *, use_rich=None: None)
    monkeypatch.setattr(sync_commands, "_confirm_or_abort", lambda _summary, *, assume_yes: None)

    def _fake_execute_pull(root: Path, dataset: str, remote_name: str, opts):
        captured["root"] = root
        captured["dataset"] = dataset
        captured["remote"] = remote_name
        captured["opts"] = opts
        return summary

    monkeypatch.setattr(sync_commands, "execute_pull", _fake_execute_pull)

    args = SimpleNamespace(
        dataset="demo",
        remote="bu-scc",
        verify="auto",
        root=root,
        rich=False,
        repo_root=None,
        remote_path=None,
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
        strict_bootstrap_id=True,
    )

    sync_commands.cmd_pull(args)

    assert captured["root"] == root
    assert captured["dataset"] == "demo"
    assert captured["remote"] == "bu-scc"
    assert getattr(captured["opts"], "verify", None) == "auto"
