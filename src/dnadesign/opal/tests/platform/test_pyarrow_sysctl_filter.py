"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/platform/test_pyarrow_sysctl_filter.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys


def test_pyarrow_sysctl_filter_install(monkeypatch) -> None:
    from dnadesign.opal.src.core import stderr_filter as sf

    calls = {"count": 0}

    class _DummyStderr:
        def isatty(self) -> bool:
            return True

    def _fake_install(_needles) -> None:
        calls["count"] += 1

    monkeypatch.setattr(sf, "_install_stderr_filter", _fake_install)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "stderr", _DummyStderr())
    monkeypatch.setenv("OPAL_SUPPRESS_PYARROW_SYSCTL", "1")
    if hasattr(sys, "_opal_stderr_filter_installed"):
        delattr(sys, "_opal_stderr_filter_installed")

    sf.maybe_install_pyarrow_sysctl_filter()
    assert calls["count"] == 1


def test_pyarrow_sysctl_filter_respects_opt_out(monkeypatch) -> None:
    from dnadesign.opal.src.core import stderr_filter as sf

    calls = {"count": 0}

    class _DummyStderr:
        def isatty(self) -> bool:
            return True

    def _fake_install(_needles) -> None:
        calls["count"] += 1

    monkeypatch.setattr(sf, "_install_stderr_filter", _fake_install)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "stderr", _DummyStderr())
    monkeypatch.setenv("OPAL_SUPPRESS_PYARROW_SYSCTL", "0")
    if hasattr(sys, "_opal_stderr_filter_installed"):
        delattr(sys, "_opal_stderr_filter_installed")

    sf.maybe_install_pyarrow_sysctl_filter()
    assert calls["count"] == 0


def test_pyarrow_sysctl_filter_default_installs_when_env_unset(monkeypatch) -> None:
    from dnadesign.opal.src.core import stderr_filter as sf

    calls = {"count": 0}

    class _DummyStderr:
        def isatty(self) -> bool:
            return False

    def _fake_install(_needles) -> None:
        calls["count"] += 1

    monkeypatch.setattr(sf, "_install_stderr_filter", _fake_install)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "stderr", _DummyStderr())
    monkeypatch.delenv("OPAL_SUPPRESS_PYARROW_SYSCTL", raising=False)
    if hasattr(sys, "_opal_stderr_filter_installed"):
        delattr(sys, "_opal_stderr_filter_installed")

    sf.maybe_install_pyarrow_sysctl_filter()
    assert calls["count"] == 1


def test_pyarrow_sysctl_filter_forces_when_not_tty(monkeypatch) -> None:
    from dnadesign.opal.src.core import stderr_filter as sf

    calls = {"count": 0}

    class _DummyStderr:
        def isatty(self) -> bool:
            return False

    def _fake_install(_needles) -> None:
        calls["count"] += 1

    monkeypatch.setattr(sf, "_install_stderr_filter", _fake_install)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "stderr", _DummyStderr())
    monkeypatch.setenv("OPAL_SUPPRESS_PYARROW_SYSCTL", "1")
    if hasattr(sys, "_opal_stderr_filter_installed"):
        delattr(sys, "_opal_stderr_filter_installed")

    sf.maybe_install_pyarrow_sysctl_filter()
    assert calls["count"] == 1


def test_pyarrow_sysctl_filter_flushes_non_matching_lines(capfd) -> None:
    from dnadesign.opal.src.core import stderr_filter as sf

    if hasattr(sys, "_opal_stderr_filter_installed"):
        delattr(sys, "_opal_stderr_filter_installed")
    if hasattr(sys, "_opal_stderr_filter_cleanup"):
        delattr(sys, "_opal_stderr_filter_cleanup")
    if hasattr(sys, "_opal_stderr_filter_cleaned"):
        delattr(sys, "_opal_stderr_filter_cleaned")

    sf._install_stderr_filter(("arrow/util/cpu_info.cc", "sysctlbyname failed for"))
    print("opal stderr passthrough", file=sys.stderr)
    sys.stderr.flush()

    cleanup = getattr(sys, "_opal_stderr_filter_cleanup", None)
    assert cleanup is not None
    cleanup()

    captured = capfd.readouterr()
    assert "opal stderr passthrough" in captured.err
