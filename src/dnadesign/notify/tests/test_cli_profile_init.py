"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_profile_init.py

Profile init command tests for notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.notify.cli import app


def _event(action: str = "write_overlay_part") -> dict:
    return {
        "event_version": 1,
        "timestamp_utc": "2026-02-06T00:00:00+00:00",
        "action": action,
        "dataset": {"name": "demo", "root": "/tmp/datasets"},
        "args": {"namespace": "densegen"},
        "metrics": {"rows_written": 3},
        "artifacts": {"overlay": {"namespace": "densegen"}},
        "fingerprint": {"rows": 1, "cols": 2, "size_bytes": 128, "sha256": None},
        "registry_hash": "abc123",
        "actor": {"tool": "densegen", "run_id": "run-1", "host": "host", "pid": 123},
        "version": "0.1.0",
    }


def _write_events(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def test_profile_init_writes_densegen_policy_defaults(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "init",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--url-env",
            "DENSEGEN_WEBHOOK",
            "--events",
            str(events),
            "--policy",
            "densegen",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["profile_version"] == 2
    assert data["provider"] == "slack"
    assert data["webhook"] == {"source": "env", "ref": "DENSEGEN_WEBHOOK"}
    assert data["policy"] == "densegen"
    assert data["only_tools"] == "densegen"
    assert "densegen_health" in data["only_actions"]
    assert "url_env" not in data


def test_profile_init_persists_progress_tunables(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "init",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--url-env",
            "DENSEGEN_WEBHOOK",
            "--events",
            str(events),
            "--progress-step-pct",
            "20",
            "--progress-min-seconds",
            "90",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["progress_step_pct"] == 20
    assert data["progress_min_seconds"] == 90.0


def test_profile_init_writes_infer_evo2_policy_defaults(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event(action="attach")])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "init",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--url-env",
            "DENSEGEN_WEBHOOK",
            "--events",
            str(events),
            "--policy",
            "infer_evo2",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["profile_version"] == 2
    assert data["webhook"] == {"source": "env", "ref": "DENSEGEN_WEBHOOK"}
    assert data["policy"] == "infer_evo2"
    assert data["only_tools"] == "infer"
    assert data["only_actions"] == "attach,materialize"


def test_profile_init_resolves_runtime_paths_from_cli_cwd(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "profiles" / "notify.profile.json"
    _write_events(events, [_event()])

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(workspace)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "init",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--url-env",
            "DENSEGEN_WEBHOOK",
            "--events",
            str(events),
            "--cursor",
            "outputs/notify/densegen/cursor",
            "--spool-dir",
            "outputs/notify/densegen/spool",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["cursor"] == str((workspace / "outputs" / "notify" / "densegen" / "cursor").resolve())
    assert data["spool_dir"] == str((workspace / "outputs" / "notify" / "densegen" / "spool").resolve())


def test_profile_init_refuses_overwrite_without_force(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])

    runner = CliRunner()
    args = [
        "profile",
        "init",
        "--profile",
        str(profile),
        "--provider",
        "slack",
        "--url-env",
        "DENSEGEN_WEBHOOK",
        "--events",
        str(events),
    ]
    first = runner.invoke(app, args)
    second = runner.invoke(app, args)
    assert first.exit_code == 0
    assert second.exit_code == 1
    assert "already exists" in second.stdout
