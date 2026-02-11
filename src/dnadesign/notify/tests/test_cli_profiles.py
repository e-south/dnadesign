"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_profiles.py

Profile workflow tests for secure, fail-fast Notify CLI setup.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnadesign.notify.cli import _ensure_private_directory, app
from dnadesign.notify.errors import NotifyConfigError


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


def test_profile_init_writes_densegen_preset(tmp_path: Path) -> None:
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
            "--preset",
            "densegen",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["provider"] == "slack"
    assert data["url_env"] == "DENSEGEN_WEBHOOK"
    assert data["only_tools"] == "densegen"
    assert "densegen_health" in data["only_actions"]
    assert "url" not in data


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


def test_profile_wizard_stores_secret_ref_when_requested(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    captured: list[tuple[str, str]] = []

    monkeypatch.setattr("dnadesign.notify.cli.is_secret_backend_available", lambda _backend: True)
    monkeypatch.setattr(
        "dnadesign.notify.cli.store_secret_ref",
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
    assert "export " not in result.stdout


def test_profile_wizard_auto_requires_secure_backend_or_explicit_env(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr("dnadesign.notify.cli.is_secret_backend_available", lambda _backend: False)

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
        ],
    )
    assert result.exit_code == 1
    assert "--secret-source env" in result.stdout


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

    monkeypatch.setattr("dnadesign.notify.cli._ensure_private_directory", _fail_private_dir)

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
    profile = tmp_path / "outputs" / "notify.profile.json"
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
            "outputs/notify.profile.json",
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


def test_profile_doctor_fails_when_webhook_env_is_missing(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 1,
                "provider": "slack",
                "url_env": "DENSEGEN_WEBHOOK",
                "events": str(events),
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 1
    assert "is not set or empty" in result.stdout


def test_profile_doctor_rejects_non_usr_events_file(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    profile = tmp_path / "notify.profile.json"
    profile.write_text(
        json.dumps(
            {
                "profile_version": 1,
                "provider": "slack",
                "url_env": "DENSEGEN_WEBHOOK",
                "events": str(cfg_path),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://example.invalid/webhook")

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 1
    assert "USR .events.log" in result.stdout


def test_profile_doctor_rejects_non_usr_events_file_v2_secret_ref(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    profile = tmp_path / "notify.profile.json"
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(cfg_path),
                "webhook": {"source": "secret_ref", "ref": "keychain://dnadesign.notify/demo"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "dnadesign.notify.validation.resolve_secret_ref", lambda _ref: "https://example.invalid/webhook"
    )

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 1
    assert "USR .events.log" in result.stdout
    assert "inspect run --usr-events-path" in result.stdout


def test_profile_doctor_passes_when_wiring_is_valid(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 1,
                "provider": "slack",
                "url_env": "DENSEGEN_WEBHOOK",
                "events": str(events),
                "cursor": str(tmp_path / "notify.cursor"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://example.invalid/webhook")

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 0
    assert "Profile wiring OK." in result.stdout


def test_profile_doctor_passes_with_secret_ref_profile(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "cursor": str(tmp_path / "notify.cursor"),
                "webhook": {"source": "secret_ref", "ref": "keychain://dnadesign.notify/demo"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "dnadesign.notify.validation.resolve_secret_ref", lambda _ref: "https://example.invalid/webhook"
    )

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 0
    assert "Profile wiring OK." in result.stdout


def test_usr_events_watch_can_use_profile_defaults(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 1,
                "provider": "generic",
                "url_env": "DENSEGEN_WEBHOOK",
                "events": str(events),
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--profile",
            str(profile),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert '"tool": "densegen"' in result.stdout


def test_usr_events_watch_can_use_profile_context_flags(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "generic",
                "events": str(events),
                "include_context": True,
                "include_raw_event": True,
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--profile",
            str(profile),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert '"host": "host"' in result.stdout
    assert '"cwd": "/tmp/datasets"' in result.stdout
    assert "usr_event" in result.stdout


def test_usr_events_watch_can_use_profile_secret_ref(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "generic",
                "events": str(events),
                "webhook": {"source": "secret_ref", "ref": "keychain://dnadesign.notify/demo"},
            }
        ),
        encoding="utf-8",
    )

    sent: list[dict] = []

    def _fake_post(_url: str, body: dict, **_kwargs) -> None:
        sent.append(body)

    monkeypatch.setattr(
        "dnadesign.notify.validation.resolve_secret_ref", lambda _ref: "https://example.invalid/webhook"
    )
    monkeypatch.setattr("dnadesign.notify.cli.post_json", _fake_post)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--profile",
            str(profile),
        ],
    )
    assert result.exit_code == 0
    assert len(sent) == 1


def test_usr_events_watch_rejects_profiles_with_plain_url(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 1,
                "provider": "generic",
                "url": "https://example.invalid/webhook",
                "events": str(events),
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--profile",
            str(profile),
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "must not store plain webhook URLs" in result.stdout


def test_spool_drain_can_use_profile_defaults(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir(parents=True, exist_ok=True)
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    payload = {
        "status": "running",
        "tool": "densegen",
        "run_id": "run-1",
        "timestamp": "2026-02-06T00:00:00+00:00",
        "host": "host",
        "cwd": "/tmp",
        "meta": {"usr_action": "write_overlay_part"},
    }
    (spool_dir / "item.json").write_text(
        json.dumps({"provider": "generic", "payload": payload, "event": _event()}),
        encoding="utf-8",
    )
    profile.write_text(
        json.dumps(
            {
                "profile_version": 1,
                "provider": "generic",
                "url_env": "DENSEGEN_WEBHOOK",
                "events": str(events),
                "spool_dir": str(spool_dir),
            }
        ),
        encoding="utf-8",
    )

    sent: list[dict] = []

    def _fake_post(_url: str, body: dict, **_kwargs) -> None:
        sent.append(body)

    monkeypatch.setattr("dnadesign.notify.cli.post_json", _fake_post)
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://example.invalid/webhook")

    runner = CliRunner()
    result = runner.invoke(app, ["spool", "drain", "--profile", str(profile)])
    assert result.exit_code == 0
    assert len(sent) == 1
    assert not list(spool_dir.glob("*.json"))


def test_usr_events_watch_resolves_profile_relative_events_path(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    events = workspace / "events.log"
    profile = workspace / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 1,
                "provider": "generic",
                "url_env": "DENSEGEN_WEBHOOK",
                "events": "events.log",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--profile",
            str(profile),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert '"tool": "densegen"' in result.stdout
