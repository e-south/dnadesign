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
