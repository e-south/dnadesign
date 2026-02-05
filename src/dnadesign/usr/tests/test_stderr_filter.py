"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_stderr_filter.py

Decision logic tests for stderr filtering behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys

import pytest

from dnadesign.usr.src import stderr_filter


@pytest.mark.parametrize(
    ("platform", "suppress", "show", "expected"),
    [
        ("darwin", None, None, True),
        ("darwin", "1", None, True),
        ("darwin", "0", None, False),
        ("darwin", None, "1", False),
        ("darwin", None, "0", True),
        ("linux", None, None, False),
    ],
)
def test_should_filter_pyarrow_sysctl(platform, suppress, show, expected, monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv("USR_SUPPRESS_PYARROW_SYSCTL", raising=False)
    monkeypatch.delenv("USR_SHOW_PYARROW_SYSCTL", raising=False)
    if suppress is not None:
        monkeypatch.setenv("USR_SUPPRESS_PYARROW_SYSCTL", suppress)
    if show is not None:
        monkeypatch.setenv("USR_SHOW_PYARROW_SYSCTL", show)

    assert stderr_filter.should_filter_pyarrow_sysctl() is expected
