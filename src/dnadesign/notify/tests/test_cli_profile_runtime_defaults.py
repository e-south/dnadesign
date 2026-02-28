"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_profile_runtime_defaults.py

Runtime profile-default tests for notify usr-events watch and spool commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
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


def _set_ssl_cert_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("dummy", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_bundle))
    return ca_bundle


def test_usr_events_watch_can_use_profile_defaults(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "generic",
                "events": str(events),
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
        "dnadesign.notify.delivery.validation.resolve_secret_ref", lambda _ref: "https://example.invalid/webhook"
    )
    monkeypatch.setattr("dnadesign.notify.cli.bindings.post_json", _fake_post)
    _set_ssl_cert_file(monkeypatch, tmp_path)

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
                "profile_version": 2,
                "provider": "generic",
                "url": "https://example.invalid/webhook",
                "events": str(events),
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
                "profile_version": 2,
                "provider": "generic",
                "events": str(events),
                "spool_dir": str(spool_dir),
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )

    sent: list[dict] = []

    def _fake_post(_url: str, body: dict, **_kwargs) -> None:
        sent.append(body)

    monkeypatch.setattr("dnadesign.notify.cli.bindings.post_json", _fake_post)
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://example.invalid/webhook")
    _set_ssl_cert_file(monkeypatch, tmp_path)

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
                "profile_version": 2,
                "provider": "generic",
                "events": "events.log",
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
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
