"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_usr_events.py

CLI adapter tests for notify usr-events watch + spool workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.notify.cli import app
from dnadesign.notify.errors import NotifyDeliveryError


def _event(*, action: str = "write_overlay_part", event_version: int = 1) -> dict:
    return {
        "event_version": event_version,
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


def test_usr_events_watch_rejects_unknown_event_version(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    _write_events(events, [_event(event_version=99)])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--provider",
            "generic",
            "--url",
            "http://example.com",
            "--events",
            str(events),
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "unknown event_version" in result.stdout


def test_usr_events_watch_rejects_non_object_actor(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    event = _event()
    event["actor"] = "densegen"
    _write_events(events, [event])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--provider",
            "generic",
            "--url",
            "http://example.com",
            "--events",
            str(events),
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "event field 'actor' must be an object" in result.stdout


def test_usr_events_watch_rejects_non_object_dataset(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    event = _event()
    event["dataset"] = "demo"
    _write_events(events, [event])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--provider",
            "generic",
            "--url",
            "http://example.com",
            "--events",
            str(events),
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "event field 'dataset' must be an object" in result.stdout


def test_usr_events_watch_dry_run_does_not_require_webhook_url(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    _write_events(events, [_event()])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--provider",
            "generic",
            "--events",
            str(events),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert '"tool": "densegen"' in result.stdout
    assert "usr_args" not in result.stdout
    assert '"usr_event":' not in result.stdout
    assert '"host"' not in result.stdout
    assert '"cwd"' not in result.stdout
    assert '"/tmp/datasets"' not in result.stdout


def test_usr_events_watch_can_skip_invalid_events_when_configured(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    bad = _event()
    bad["actor"] = "densegen"
    good = _event()
    _write_events(events, [bad, good])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--provider",
            "generic",
            "--events",
            str(events),
            "--dry-run",
            "--on-invalid-event",
            "skip",
        ],
    )
    assert result.exit_code == 0
    assert "skipping invalid event line" in result.stdout.lower()
    assert '"tool": "densegen"' in result.stdout


def test_usr_events_watch_cursor_prevents_duplicate_sends(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    cursor = tmp_path / "cursor.txt"
    _write_events(events, [_event()])

    sent_payloads: list[dict] = []

    def _fake_post(_url: str, payload: dict, **_kwargs) -> None:
        sent_payloads.append(payload)

    monkeypatch.setattr("dnadesign.notify.cli.post_json", _fake_post)

    runner = CliRunner()
    args = [
        "usr-events",
        "watch",
        "--provider",
        "generic",
        "--url",
        "http://example.com",
        "--events",
        str(events),
        "--cursor",
        str(cursor),
    ]
    result_first = runner.invoke(app, args)
    result_second = runner.invoke(app, args)

    assert result_first.exit_code == 0
    assert result_second.exit_code == 0
    assert len(sent_payloads) == 1


def test_usr_events_watch_requires_actor_tool_or_override(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    event = _event()
    event["actor"] = {"run_id": "run-1", "host": "host", "pid": 123}
    _write_events(events, [event])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--provider",
            "generic",
            "--url",
            "http://example.com",
            "--events",
            str(events),
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "event missing actor.tool" in result.stdout


def test_usr_events_watch_requires_actor_run_id_or_override(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    event = _event()
    event["actor"] = {"tool": "densegen", "host": "host", "pid": 123}
    _write_events(events, [event])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--provider",
            "generic",
            "--url",
            "http://example.com",
            "--events",
            str(events),
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "event missing actor.run_id" in result.stdout


def test_usr_events_watch_fails_on_truncated_cursor_by_default(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    cursor = tmp_path / "cursor.txt"
    _write_events(events, [_event()])
    cursor.write_text("99999", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--provider",
            "generic",
            "--url",
            "http://example.com",
            "--events",
            str(events),
            "--cursor",
            str(cursor),
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "cursor offset exceeds events file size" in result.stdout


def test_usr_events_watch_can_restart_on_truncated_cursor_when_configured(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    cursor = tmp_path / "cursor.txt"
    _write_events(events, [_event()])
    cursor.write_text("99999", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--provider",
            "generic",
            "--url",
            "http://example.com",
            "--events",
            str(events),
            "--cursor",
            str(cursor),
            "--on-truncate",
            "restart",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert cursor.read_text(encoding="utf-8").strip() != "99999"


def test_usr_events_watch_spools_after_delivery_failures(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    spool_dir = tmp_path / "spool"
    _write_events(events, [_event(action="densegen_flush_failed")])

    calls = {"n": 0}

    def _always_fail(_url: str, _payload: dict, **_kwargs) -> None:
        calls["n"] += 1
        raise NotifyDeliveryError("network down")

    monkeypatch.setattr("dnadesign.notify.cli.post_json", _always_fail)
    monkeypatch.setattr("dnadesign.notify.cli.time.sleep", lambda *_args, **_kwargs: None)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--provider",
            "generic",
            "--url",
            "http://example.com",
            "--events",
            str(events),
            "--spool-dir",
            str(spool_dir),
            "--retry-max",
            "2",
            "--retry-base-seconds",
            "0.0",
        ],
    )
    assert result.exit_code == 0
    assert calls["n"] >= 2
    assert list(spool_dir.glob("*.json"))
    spooled = next(iter(spool_dir.glob("*.json")))
    assert (spooled.stat().st_mode & 0o777) == 0o600
    assert (spool_dir.stat().st_mode & 0o777) == 0o700


def test_spool_drain_sends_and_deletes_successful_files(tmp_path: Path, monkeypatch) -> None:
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "running",
        "tool": "densegen",
        "run_id": "run-1",
        "timestamp": "2026-02-06T00:00:00+00:00",
        "host": "host",
        "cwd": "/tmp",
        "meta": {"usr_action": "write_overlay_part"},
    }
    spool_file = spool_dir / "item.json"
    spool_file.write_text(json.dumps({"provider": "generic", "payload": payload, "event": _event()}), encoding="utf-8")

    sent: list[dict] = []

    def _fake_post(_url: str, body: dict, **_kwargs) -> None:
        sent.append(body)

    monkeypatch.setattr("dnadesign.notify.cli.post_json", _fake_post)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "spool",
            "drain",
            "--spool-dir",
            str(spool_dir),
            "--provider",
            "generic",
            "--url",
            "http://example.com",
        ],
    )
    assert result.exit_code == 0
    assert len(sent) == 1
    assert not spool_file.exists()


def test_usr_events_watch_can_include_args_when_opted_in(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    _write_events(events, [_event()])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--provider",
            "generic",
            "--events",
            str(events),
            "--include-args",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "usr_args" in result.stdout


def test_usr_events_watch_can_include_raw_event_when_opted_in(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    _write_events(events, [_event()])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--provider",
            "generic",
            "--events",
            str(events),
            "--include-raw-event",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "usr_event" in result.stdout


def test_usr_events_watch_can_include_context_when_opted_in(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    _write_events(events, [_event()])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--provider",
            "generic",
            "--events",
            str(events),
            "--include-context",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert '"host": "host"' in result.stdout
    assert '"cwd": "/tmp/datasets"' in result.stdout
    assert '"usr_dataset_root": "/tmp/datasets"' in result.stdout


def test_usr_events_watch_formats_densegen_health_message(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    event = _event(action="densegen_health")
    event["args"] = {
        "status": "running",
        "plan": "demo_plan",
        "input_name": "pool_a",
        "sampling_library_index": 2,
    }
    event["metrics"] = {
        "rows_written_session": 12,
        "run_quota": 120,
        "quota_progress_pct": 10.0,
        "compression_ratio": 1.13,
        "flush_count": 3,
    }
    _write_events(events, [event])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--provider",
            "generic",
            "--events",
            str(events),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "densegen running on demo" in result.stdout
    assert "rows=12/120" in result.stdout
    assert "quota=10.0%" in result.stdout


def test_usr_events_watch_maps_densegen_health_completed_to_success_status(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    event = _event(action="densegen_health")
    event["args"] = {"status": "completed"}
    _write_events(events, [event])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--provider",
            "generic",
            "--events",
            str(events),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert '"status": "success"' in result.stdout
