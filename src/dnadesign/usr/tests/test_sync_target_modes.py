"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_sync_target_modes.py

Tests for sync target mode classification and dataset directory resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
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
        no_verify_sidecars=True,
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


def test_cmd_pull_dataset_defaults_to_hash_and_strict_sidecars(tmp_path: Path, monkeypatch) -> None:
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
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path=None,
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
    )

    sync_commands.cmd_pull(args)

    assert getattr(captured["opts"], "verify", None) == "hash"
    assert getattr(captured["opts"], "verify_sidecars", None) is True
    assert getattr(captured["opts"], "verify_derived_hashes", None) is True


def test_cmd_pull_dataset_supports_no_verify_sidecars_opt_out(tmp_path: Path, monkeypatch) -> None:
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
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path=None,
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
        no_verify_sidecars=True,
    )

    sync_commands.cmd_pull(args)

    assert getattr(captured["opts"], "verify", None) == "hash"
    assert getattr(captured["opts"], "verify_sidecars", None) is False
    assert getattr(captured["opts"], "verify_derived_hashes", None) is False


def test_cmd_pull_dataset_supports_no_verify_derived_hashes_opt_out(tmp_path: Path, monkeypatch) -> None:
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
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path=None,
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
        no_verify_derived_hashes=True,
    )

    sync_commands.cmd_pull(args)

    assert getattr(captured["opts"], "verify", None) == "hash"
    assert getattr(captured["opts"], "verify_sidecars", None) is True
    assert getattr(captured["opts"], "verify_derived_hashes", None) is False


def test_cmd_pull_file_mode_defaults_to_hash_and_sidecars_off(tmp_path: Path, monkeypatch) -> None:
    file_target = tmp_path / "records.parquet"
    file_target.write_text("stub", encoding="utf-8")

    summary = SimpleNamespace(has_change=False, verify_notes=[])
    captured: dict[str, object] = {}

    monkeypatch.setattr(sync_commands, "plan_diff_file", lambda *_args, **_kwargs: summary)
    monkeypatch.setattr(sync_commands, "_print_verify_notes", lambda _summary: None)
    monkeypatch.setattr(sync_commands, "_print_diff", lambda _summary, *, use_rich=None: None)
    monkeypatch.setattr(sync_commands, "_confirm_or_abort", lambda _summary, *, assume_yes: None)
    monkeypatch.setattr(sync_commands, "_resolve_remote_path_for_file", lambda *_args, **_kwargs: "/remote/records")

    def _fake_execute_pull_file(local_file: Path, remote_name: str, remote_path: str, opts):
        captured["local_file"] = local_file
        captured["remote"] = remote_name
        captured["remote_path"] = remote_path
        captured["opts"] = opts
        return summary

    monkeypatch.setattr(sync_commands, "execute_pull_file", _fake_execute_pull_file)

    args = SimpleNamespace(
        dataset=str(file_target),
        remote="bu-scc",
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path=None,
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
    )

    sync_commands.cmd_pull(args)

    assert getattr(captured["opts"], "verify", None) == "hash"
    assert getattr(captured["opts"], "verify_sidecars", None) is False


def test_cmd_pull_rejects_conflicting_sidecar_flags(tmp_path: Path) -> None:
    args = SimpleNamespace(
        dataset="densegen/demo_hpc_remote_only",
        remote="bu-scc",
        verify="hash",
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path=None,
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
        verify_sidecars=True,
        no_verify_sidecars=True,
    )

    try:
        sync_commands.cmd_pull(args)
    except SystemExit as exc:
        assert "Cannot combine --verify-sidecars and --no-verify-sidecars" in str(exc)
        return
    raise AssertionError("expected conflicting sidecar flags to fail fast")


def test_cmd_pull_passes_verify_derived_hashes_option(tmp_path: Path, monkeypatch) -> None:
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
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path=None,
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
        verify_derived_hashes=True,
    )

    sync_commands.cmd_pull(args)

    assert getattr(captured["opts"], "verify", None) == "hash"
    assert getattr(captured["opts"], "verify_sidecars", None) is True
    assert getattr(captured["opts"], "verify_derived_hashes", None) is True


def test_cmd_pull_rejects_verify_derived_hashes_with_no_verify_sidecars(tmp_path: Path) -> None:
    args = SimpleNamespace(
        dataset="densegen/demo_hpc_remote_only",
        remote="bu-scc",
        verify="hash",
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path=None,
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
        no_verify_sidecars=True,
        verify_derived_hashes=True,
    )

    try:
        sync_commands.cmd_pull(args)
    except SystemExit as exc:
        assert "requires sidecar verification" in str(exc)
        return
    raise AssertionError("expected verify-derived-hashes with no-verify-sidecars to fail fast")


def test_cmd_pull_rejects_conflicting_derived_hash_flags(tmp_path: Path) -> None:
    args = SimpleNamespace(
        dataset="densegen/demo_hpc_remote_only",
        remote="bu-scc",
        verify="hash",
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path=None,
        primary_only=False,
        skip_snapshots=False,
        dry_run=False,
        yes=True,
        verify_derived_hashes=True,
        no_verify_derived_hashes=True,
    )

    try:
        sync_commands.cmd_pull(args)
    except SystemExit as exc:
        assert "Cannot combine --verify-derived-hashes and --no-verify-derived-hashes" in str(exc)
        return
    raise AssertionError("expected conflicting derived-hash flags to fail fast")


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
        lambda summary, *, action, dry_run, verify_sidecars, verify_derived_hashes: captured.update(
            {"summary": summary, "action": action}
        ),
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
        lambda summary, *, action, dry_run, verify_sidecars, verify_derived_hashes: captured.update(
            {"summary": summary, "action": action}
        ),
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


def test_cmd_pull_writes_sync_audit_json_artifact(tmp_path: Path, monkeypatch) -> None:
    audit_path = tmp_path / "audit" / "pull.json"
    summary = SimpleNamespace(
        dataset="densegen/demo",
        has_change=True,
        verify_mode="hash",
        changes={
            "primary_sha_diff": True,
            "meta_mtime_diff": False,
            "snapshots_name_diff": False,
            "derived_files_diff": True,
            "aux_files_diff": False,
        },
        events_local_lines=2,
        events_remote_lines=4,
        snapshots=SimpleNamespace(count=3, newer_than_local=1),
        derived_local_files=["densegen/part-001.parquet"],
        derived_remote_files=["densegen/part-001.parquet", "densegen/part-002.parquet"],
        aux_local_files=["_artifacts/a.json"],
        aux_remote_files=["_artifacts/a.json"],
    )
    monkeypatch.setattr(sync_commands, "_is_file_mode_target", lambda _target: False)
    monkeypatch.setattr(
        sync_commands,
        "_run_dataset_sync",
        lambda _args, *, action, resolve_target, execute_dataset: sync_commands.sync_execution_commands.SyncRunResult(
            summary=summary,
            verify_sidecars=True,
            verify_derived_hashes=True,
        ),
    )

    args = SimpleNamespace(
        dataset="densegen/demo",
        remote="bu-scc",
        dry_run=False,
        audit_json_out=str(audit_path),
    )
    sync_commands.cmd_pull(args)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["usr_output_version"] == sync_commands.USR_OUTPUT_VERSION
    assert payload["data"]["action"] == "pull"
    assert payload["data"]["verify"]["content_hashes"] == "on"
    assert payload["data"]["_derived"]["changed"] is True


def test_cmd_push_writes_sync_audit_json_artifact(tmp_path: Path, monkeypatch) -> None:
    audit_path = tmp_path / "audit" / "push.json"
    summary = SimpleNamespace(
        dataset="densegen/demo",
        has_change=False,
        verify_mode="hash",
        changes={
            "primary_sha_diff": False,
            "meta_mtime_diff": False,
            "snapshots_name_diff": False,
            "derived_files_diff": False,
            "aux_files_diff": False,
        },
        events_local_lines=5,
        events_remote_lines=5,
        snapshots=SimpleNamespace(count=2, newer_than_local=0),
        derived_local_files=[],
        derived_remote_files=[],
        aux_local_files=[],
        aux_remote_files=[],
    )
    monkeypatch.setattr(sync_commands, "_is_file_mode_target", lambda _target: False)
    monkeypatch.setattr(
        sync_commands,
        "_run_dataset_sync",
        lambda _args, *, action, resolve_target, execute_dataset: sync_commands.sync_execution_commands.SyncRunResult(
            summary=summary,
            verify_sidecars=False,
            verify_derived_hashes=False,
        ),
    )

    args = SimpleNamespace(
        dataset="densegen/demo",
        remote="bu-scc",
        dry_run=False,
        audit_json_out=str(audit_path),
    )
    sync_commands.cmd_push(args)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["usr_output_version"] == sync_commands.USR_OUTPUT_VERSION
    assert payload["data"]["action"] == "push"
    assert payload["data"]["transfer_state"] == "NO-OP"
    assert payload["data"]["verify"]["sidecars"] == "off"


def test_cmd_diff_writes_sync_audit_json_artifact(tmp_path: Path, monkeypatch) -> None:
    audit_path = tmp_path / "audit" / "diff.json"
    summary = SimpleNamespace(
        dataset="densegen/demo",
        has_change=True,
        verify_mode="hash",
        changes={
            "primary_sha_diff": True,
            "meta_mtime_diff": True,
            "snapshots_name_diff": False,
            "derived_files_diff": True,
            "aux_files_diff": True,
        },
        events_local_lines=1,
        events_remote_lines=3,
        snapshots=SimpleNamespace(count=2, newer_than_local=1),
        derived_local_files=["densegen/part-001.parquet"],
        derived_remote_files=["densegen/part-001.parquet", "densegen/part-002.parquet"],
        aux_local_files=["_registry/a.yaml"],
        aux_remote_files=["_registry/a.yaml", "_registry/b.yaml"],
        primary_local=SimpleNamespace(sha256="a", size=1, rows=1, cols=1),
        primary_remote=SimpleNamespace(sha256="b", size=2, rows=2, cols=1),
        verify_notes=[],
        meta_local_mtime="1",
        meta_remote_mtime="2",
    )
    monkeypatch.setattr(sync_commands, "_is_file_mode_target", lambda _target: False)
    monkeypatch.setattr(sync_commands, "_is_dataset_dir_target", lambda _target: False)
    monkeypatch.setattr(
        sync_commands,
        "_resolve_dataset_id_for_diff_or_pull",
        lambda _root, _dataset, *, use_rich: "densegen/demo",
    )
    monkeypatch.setattr(sync_commands, "plan_diff", lambda *_args, **_kwargs: summary)
    monkeypatch.setattr(sync_commands, "_print_verify_notes", lambda _summary: None)
    monkeypatch.setattr(sync_commands, "_print_diff", lambda _summary, *, use_rich=None: None)

    args = SimpleNamespace(
        dataset="densegen/demo",
        remote="bu-scc",
        verify="hash",
        root=tmp_path / "usr_root",
        rich=False,
        repo_root=None,
        remote_path=None,
        format="plain",
        audit_json_out=str(audit_path),
    )
    sync_commands.cmd_diff(
        args,
        resolve_output_format=lambda _args: "plain",
        print_json=lambda _payload: None,
        output_version=sync_commands.USR_OUTPUT_VERSION,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["usr_output_version"] == sync_commands.USR_OUTPUT_VERSION
    assert payload["data"]["action"] == "diff"
    assert payload["data"]["_derived"]["changed"] is True
    assert payload["data"]["_auxiliary"]["changed"] is True
    assert payload["data"][".events.log"]["remote"] == 3


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


def test_cmd_pull_file_mode_rejects_no_verify_derived_hashes(tmp_path: Path) -> None:
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
        no_verify_derived_hashes=True,
    )

    try:
        sync_commands.cmd_pull(args)
    except SystemExit as exc:
        assert "dataset-only flags" in str(exc)
        return
    raise AssertionError("expected no-verify-derived-hashes in FILE mode to fail fast")


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
