"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/cli/test_console.py

CLI console behavior contracts for infer.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.infer.src.cli import console as console_mod


def test_console_setup_logging_uses_shared_policy_owner(monkeypatch) -> None:
    calls = []

    def _fake_setup(level: str = "INFO", json_logs: bool = False, *, rich_console=None) -> None:
        calls.append((level, json_logs, rich_console))

    monkeypatch.setattr("dnadesign.infer.src.cli.console._shared_setup_console_logging", _fake_setup)

    console_mod.setup_console_logging("DEBUG", True)

    assert calls == [("DEBUG", True, console_mod.console)]
