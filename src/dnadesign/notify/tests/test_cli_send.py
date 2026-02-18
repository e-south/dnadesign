"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_send.py

CLI behavior tests for the notify send command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typer.testing import CliRunner

from dnadesign.notify.cli import app


def test_notify_send_dry_run_allows_https_without_ca_bundle(monkeypatch) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "send",
            "--status",
            "started",
            "--tool",
            "densegen",
            "--run-id",
            "run-1",
            "--provider",
            "generic",
            "--url",
            "https://example.invalid/webhook",
            "--message",
            "smoke",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert '"run_id": "run-1"' in result.stdout
    assert '"status": "started"' in result.stdout


def test_notify_send_rejects_https_without_ca_bundle_when_not_dry_run(monkeypatch) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "send",
            "--status",
            "started",
            "--tool",
            "densegen",
            "--run-id",
            "run-1",
            "--provider",
            "generic",
            "--url",
            "https://example.invalid/webhook",
        ],
    )
    assert result.exit_code == 1
    assert "requires an explicit CA bundle" in result.stdout
