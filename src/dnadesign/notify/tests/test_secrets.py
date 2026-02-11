"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_secrets.py

Tests for runtime availability checks of notify secret backends.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess

from dnadesign.notify.secrets import is_secret_backend_available


def test_secretservice_unavailable_without_dbus_session(monkeypatch) -> None:
    monkeypatch.setattr("dnadesign.notify.secrets.shutil.which", lambda _cmd: "/usr/bin/secret-tool")
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)

    assert is_secret_backend_available("secretservice") is False


def test_secretservice_available_with_probeable_runtime(monkeypatch) -> None:
    monkeypatch.setattr("dnadesign.notify.secrets.shutil.which", lambda _cmd: "/usr/bin/secret-tool")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/tmp/dbus-test")
    monkeypatch.setattr(
        "dnadesign.notify.secrets.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=""),
    )

    assert is_secret_backend_available("secretservice") is True


def test_keychain_availability_uses_runtime_probe(monkeypatch) -> None:
    monkeypatch.setattr("dnadesign.notify.secrets.shutil.which", lambda _cmd: "/usr/bin/security")
    monkeypatch.setattr(
        "dnadesign.notify.secrets.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(args=[], returncode=44, stdout="", stderr="not found"),
    )

    assert is_secret_backend_available("keychain") is True

