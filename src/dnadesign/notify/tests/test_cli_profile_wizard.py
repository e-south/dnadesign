"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_profile_wizard.py

Profile wizard command tests for notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnadesign.notify.cli import app
from dnadesign.notify.errors import NotifyConfigError
from dnadesign.notify.runtime.spool import ensure_private_directory as _ensure_private_directory


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


def test_profile_wizard_writes_generic_policy_without_filters(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event(action="attach")])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(events),
            "--policy",
            "generic",
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["policy"] == "generic"
    assert "only_actions" not in data
    assert "only_tools" not in data


def test_profile_wizard_writes_v2_profile_with_env_source(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(events),
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["profile_version"] == 2
    assert data["provider"] == "slack"
    assert data["events"] == str(events)
    assert data["webhook"] == {"source": "env", "ref": "DENSEGEN_WEBHOOK"}
    assert "url_env" not in data
    assert "Next steps:" in result.stdout
    assert "notify profile doctor --profile" in result.stdout
    assert "notify usr-events watch --profile" in result.stdout
    assert "--dry-run" in result.stdout
    assert "export DENSEGEN_WEBHOOK" in result.stdout
    assert "NOTIFY_PROFILE" in result.stdout
    assert "docs/bu-scc/jobs/notify-watch.qsub" in result.stdout


def test_profile_wizard_persists_progress_tunables(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(events),
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
            "--progress-step-pct",
            "12",
            "--progress-min-seconds",
            "75",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["progress_step_pct"] == 12
    assert data["progress_min_seconds"] == 75.0


def test_profile_wizard_env_uses_default_webhook_env_when_not_set(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(events),
            "--secret-source",
            "env",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["webhook"] == {"source": "env", "ref": "NOTIFY_WEBHOOK"}


def test_profile_wizard_defaults_to_namespaced_profile_and_runtime_paths(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    _write_events(events, [_event()])
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--provider",
            "slack",
            "--events",
            str(events),
            "--policy",
            "densegen",
            "--secret-source",
            "env",
        ],
    )
    assert result.exit_code == 0
    profile = tmp_path / "outputs" / "notify" / "densegen" / "profile.json"
    assert profile.exists()
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["webhook"] == {"source": "env", "ref": "NOTIFY_WEBHOOK"}
    assert data["cursor"] == str((tmp_path / "outputs" / "notify" / "densegen" / "cursor").resolve())
    assert data["spool_dir"] == str((tmp_path / "outputs" / "notify" / "densegen" / "spool").resolve())


def test_profile_wizard_requires_policy_or_profile_for_default_events_mode_profile(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    _write_events(events, [_event()])
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--provider",
            "slack",
            "--events",
            str(events),
            "--secret-source",
            "env",
        ],
    )
    assert result.exit_code == 1
    assert "pass --policy or --profile" in result.stdout.lower()


def test_profile_wizard_stores_tls_ca_bundle_path(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("dummy", encoding="utf-8")
    _write_events(events, [_event()])
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--provider",
            "slack",
            "--events",
            str(events),
            "--policy",
            "densegen",
            "--secret-source",
            "env",
            "--tls-ca-bundle",
            str(ca_bundle),
        ],
    )
    assert result.exit_code == 0
    profile = tmp_path / "outputs" / "notify" / "densegen" / "profile.json"
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["tls_ca_bundle"] == str(ca_bundle.resolve())


def test_profile_wizard_can_emit_json_output(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(events),
            "--policy",
            "densegen",
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["profile"] == str(profile.resolve())
    assert payload["events"] == str(events)
    assert payload["events_exists"] is True
    assert payload["policy"] == "densegen"
    assert payload["webhook"] == {"source": "env", "ref": "DENSEGEN_WEBHOOK"}
    assert isinstance(payload.get("next_steps"), list)


def test_profile_wizard_stores_secret_ref_when_requested(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    captured: list[tuple[str, str]] = []

    monkeypatch.setattr("dnadesign.notify.cli.bindings.is_secret_backend_available", lambda _backend: True)
    monkeypatch.setattr(
        "dnadesign.notify.cli.bindings.store_secret_ref",
        lambda secret_ref, secret_value: captured.append((secret_ref, secret_value)),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(events),
            "--secret-source",
            "keychain",
            "--secret-ref",
            "keychain://dnadesign.notify/demo",
            "--webhook-url",
            "https://example.invalid/webhook",
        ],
    )
    assert result.exit_code == 0
    assert captured == [("keychain://dnadesign.notify/demo", "https://example.invalid/webhook")]
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["webhook"] == {"source": "secret_ref", "ref": "keychain://dnadesign.notify/demo"}
    assert "Next steps:" in result.stdout
    assert "notify profile doctor --profile" in result.stdout
    assert "notify usr-events watch --profile" in result.stdout
    assert "NOTIFY_PROFILE" in result.stdout
    assert "docs/bu-scc/jobs/notify-watch.qsub" in result.stdout
    assert "export " not in result.stdout


def test_profile_wizard_auto_falls_back_to_file_secret_ref(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr("dnadesign.notify.cli.bindings.is_secret_backend_available", lambda backend: backend == "file")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(events),
            "--webhook-url",
            "https://example.invalid/webhook",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["webhook"]["source"] == "secret_ref"
    assert data["webhook"]["ref"].startswith("file://")


def test_profile_wizard_rejects_yaml_path_for_events(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    profile = tmp_path / "notify.profile.json"
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(cfg_path),
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
        ],
    )
    assert result.exit_code == 1
    assert "USR .events.log" in result.stdout
    assert "inspect run --usr-events-path" in result.stdout


def test_profile_wizard_reports_actionable_cursor_directory_error(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    def _fail_private_dir(path: Path, *, label: str) -> None:
        raise NotifyConfigError(f"failed to set secure permissions on {label}: {path}")

    monkeypatch.setattr("dnadesign.notify.cli.bindings._ensure_private_directory", _fail_private_dir)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(events),
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
        ],
    )
    assert result.exit_code == 1
    assert "--cursor" in result.stdout
    assert "--spool-dir" in result.stdout


def test_ensure_private_directory_wraps_mkdir_permission_errors(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "restricted" / "dir"

    def _deny_mkdir(self, *args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "mkdir", _deny_mkdir)

    with pytest.raises(NotifyConfigError, match="failed to create cursor directory"):
        _ensure_private_directory(target, label="cursor directory")


def test_profile_wizard_stores_absolute_events_path(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "outputs" / "notify" / "generic" / "profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            "outputs/notify/generic/profile.json",
            "--provider",
            "slack",
            "--events",
            "events.log",
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["events"] == str(events.resolve())
