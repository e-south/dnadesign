"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_profile_smoke.py

CLI tests for notify profile smoke command orchestration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.notify.cli import app


def test_profile_smoke_runs_setup_doctor_watch_in_order(tmp_path: Path, monkeypatch) -> None:
    profile_path = tmp_path / "profile.json"
    config_path = tmp_path / "config.yaml"
    cursor_path = tmp_path / "cursor"
    spool_dir = tmp_path / "spool"
    ca_bundle = tmp_path / "ca.pem"
    events_path = tmp_path / "outputs" / "usr_datasets" / "demo" / ".events.log"
    calls: list[str] = []

    profile_path.write_text("{}", encoding="utf-8")
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    ca_bundle.write_text("dummy", encoding="utf-8")

    def _fake_setup_slack_command(**_kwargs) -> None:
        calls.append("setup")

    def _fake_profile_doctor_command(**_kwargs) -> None:
        calls.append("doctor")

    def _fake_usr_events_watch_command(**_kwargs) -> None:
        calls.append("watch")

    def _fake_resolve_tool_events_path(*, tool: str, config: Path) -> tuple[Path, str | None]:
        assert tool == "densegen"
        assert Path(config) == config_path
        return events_path, "densegen"

    monkeypatch.setattr("dnadesign.notify.cli.bindings.run_setup_slack_command", _fake_setup_slack_command)
    monkeypatch.setattr("dnadesign.notify.cli.bindings.run_profile_doctor_command", _fake_profile_doctor_command)
    monkeypatch.setattr("dnadesign.notify.cli.bindings.run_usr_events_watch_command", _fake_usr_events_watch_command)
    monkeypatch.setattr("dnadesign.notify.cli.bindings._resolve_tool_events_path", _fake_resolve_tool_events_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "smoke",
            "--profile",
            str(profile_path),
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--cursor",
            str(cursor_path),
            "--spool-dir",
            str(spool_dir),
            "--policy",
            "densegen",
            "--secret-source",
            "file",
            "--secret-ref",
            "file:///tmp/notify.secret",
            "--tls-ca-bundle",
            str(ca_bundle),
            "--only-tools",
            "densegen",
            "--dry-run",
            "--no-advance-cursor-on-dry-run",
        ],
    )

    assert result.exit_code == 0
    assert calls == ["setup", "doctor", "watch"]
    assert events_path.exists()
