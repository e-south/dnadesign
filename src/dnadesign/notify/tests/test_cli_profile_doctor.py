"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_profile_doctor.py

Profile doctor command tests for notify CLI.

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


def test_profile_doctor_fails_when_webhook_env_is_missing(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
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
                "profile_version": 2,
                "provider": "slack",
                "events": str(cfg_path),
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://hooks.slack.com/services/T000/B000/XXX")

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
        "dnadesign.notify.delivery.validation.resolve_secret_ref", lambda _ref: "https://example.invalid/webhook"
    )

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 1
    assert "USR .events.log" in result.stdout
    assert "inspect run --usr-events-path" in result.stdout


def test_profile_doctor_rejects_legacy_profile_version_1(tmp_path: Path, monkeypatch) -> None:
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
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://hooks.slack.com/services/T000/B000/XXX")

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 1
    assert "profile_version must be 2" in result.stdout


def test_profile_doctor_rejects_legacy_preset_field(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
                "preset": "densegen",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://hooks.slack.com/services/T000/B000/XXX")

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 1
    assert "legacy profile field 'preset' is not supported" in result.stdout


def test_profile_doctor_rejects_policy_without_explicit_filters(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "policy": "densegen",
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://hooks.slack.com/services/T000/B000/XXX")

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 1
    assert "requires explicit only_actions and only_tools" in result.stdout


def test_profile_doctor_passes_when_wiring_is_valid(tmp_path: Path, monkeypatch) -> None:
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
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://hooks.slack.com/services/T000/B000/XXX")
    _set_ssl_cert_file(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 0
    assert "Profile wiring OK." in result.stdout


def test_profile_doctor_can_emit_json_output(tmp_path: Path, monkeypatch) -> None:
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
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://hooks.slack.com/services/T000/B000/XXX")
    _set_ssl_cert_file(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["profile"] == str(profile.resolve())
    assert payload["events"] == str(events)


def test_profile_doctor_json_reports_structured_error(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "is not set or empty" in str(payload["error"])


def test_profile_doctor_rejects_non_slack_webhook_for_slack_provider(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://example.invalid/webhook")
    _set_ssl_cert_file(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 1
    assert "slack provider requires webhook host" in result.stdout


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
        "dnadesign.notify.delivery.validation.resolve_secret_ref",
        lambda _ref: "https://hooks.slack.com/services/T000/B000/XXX",
    )
    _set_ssl_cert_file(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 0
    assert "Profile wiring OK." in result.stdout


def test_profile_doctor_allows_missing_events_when_profile_has_events_source(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "future" / ".events.log"
    config_path = tmp_path / "densegen.config.yaml"
    profile = tmp_path / "notify.profile.json"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "events_source": {"tool": "densegen", "config": str(config_path)},
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://hooks.slack.com/services/T000/B000/XXX")
    _set_ssl_cert_file(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 0
    assert "events file not created yet" in result.stdout


def test_profile_doctor_json_reports_missing_events_as_pending_for_events_source(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "future" / ".events.log"
    config_path = tmp_path / "densegen.config.yaml"
    profile = tmp_path / "notify.profile.json"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "events_source": {"tool": "densegen", "config": str(config_path)},
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://hooks.slack.com/services/T000/B000/XXX")
    _set_ssl_cert_file(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["events_exists"] is False
    assert payload["events"] == str(events)
