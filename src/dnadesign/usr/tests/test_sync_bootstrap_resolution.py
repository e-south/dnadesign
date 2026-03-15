"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_sync_bootstrap_resolution.py

Tests for sync bootstrap dataset-id resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import dnadesign.usr.src.cli_commands.sync as sync_commands


def test_cmd_pull_allows_explicit_missing_unqualified_dataset_id_for_bootstrap(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "usr_root"
    root.mkdir(parents=True)

    summary = SimpleNamespace(has_change=False, verify_notes=[])
    captured: dict[str, object] = {}

    monkeypatch.setattr(sync_commands, "plan_diff", lambda *_args, **_kwargs: summary)
    monkeypatch.setattr(sync_commands, "_print_verify_notes", lambda _summary: None)
    monkeypatch.setattr(sync_commands, "_print_diff", lambda _summary, *, use_rich=None: None)
    monkeypatch.setattr(sync_commands, "_confirm_or_abort", lambda _summary, *, assume_yes: None)

    def _fake_execute_pull(resolved_root: Path, dataset: str, remote_name: str, opts):
        captured["root"] = resolved_root
        captured["dataset"] = dataset
        captured["remote"] = remote_name
        captured["opts"] = opts
        return summary

    monkeypatch.setattr(sync_commands, "execute_pull", _fake_execute_pull)

    args = SimpleNamespace(
        dataset="demo_remote_only",
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
        strict_bootstrap_id=False,
        verify_sidecars=False,
        no_verify_sidecars=False,
        verify_derived_hashes=False,
        no_verify_derived_hashes=False,
        audit_json_out=None,
    )

    sync_commands.cmd_pull(args)

    assert captured["root"] == root
    assert captured["dataset"] == "demo_remote_only"
    assert captured["remote"] == "bu-scc"
    assert getattr(captured["opts"], "verify", None) == "auto"
