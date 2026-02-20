"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/cli/test_cli_common.py

Tests shared CLI command helpers for error-report stream behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.opal.src.cli.commands import _common
from dnadesign.opal.src.core.utils import OpalError


class _NonInteractiveStderr:
    def isatty(self) -> bool:
        return False


def test_opal_error_default_mode_writes_only_stderr(monkeypatch) -> None:
    stderr_messages: list[str] = []
    stdout_messages: list[str] = []
    monkeypatch.delenv("OPAL_DEBUG", raising=False)
    monkeypatch.setattr(_common, "print_stderr", lambda s: stderr_messages.append(str(s)))
    monkeypatch.setattr(_common, "print_stdout", lambda s: stdout_messages.append(str(s)))
    monkeypatch.setattr(_common.sys, "stderr", _NonInteractiveStderr())

    _common.opal_error("validate", OpalError("bad config"))

    assert stderr_messages == ["bad config"]
    assert stdout_messages == []
