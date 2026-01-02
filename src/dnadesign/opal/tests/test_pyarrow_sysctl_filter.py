"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_pyarrow_sysctl_filter.py

Regression guard: ensure the macOS PyArrow sysctlbyname filter is installable
and respects the opt-out flag.

Module Author(s): Eric J. South (extended by Codex)
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys


def test_pyarrow_sysctl_filter_install(monkeypatch) -> None:
    from dnadesign.opal.src.core import stderr_filter as sf

    calls = {"count": 0}

    def _fake_install(_needles) -> None:
        calls["count"] += 1

    monkeypatch.setattr(sf, "_install_stderr_filter", _fake_install)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("OPAL_SUPPRESS_PYARROW_SYSCTL", "1")
    if hasattr(sys, "_opal_stderr_filter_installed"):
        delattr(sys, "_opal_stderr_filter_installed")

    sf.maybe_install_pyarrow_sysctl_filter()
    assert calls["count"] == 1


def test_pyarrow_sysctl_filter_respects_opt_out(monkeypatch) -> None:
    from dnadesign.opal.src.core import stderr_filter as sf

    calls = {"count": 0}

    def _fake_install(_needles) -> None:
        calls["count"] += 1

    monkeypatch.setattr(sf, "_install_stderr_filter", _fake_install)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("OPAL_SUPPRESS_PYARROW_SYSCTL", "0")
    if hasattr(sys, "_opal_stderr_filter_installed"):
        delattr(sys, "_opal_stderr_filter_installed")

    sf.maybe_install_pyarrow_sysctl_filter()
    assert calls["count"] == 0
