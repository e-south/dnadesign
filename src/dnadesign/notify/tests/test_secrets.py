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

from dnadesign.notify.secrets import (
    is_secret_backend_available,
    resolve_secret_ref,
    store_secret_ref,
)


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


def test_secretservice_available_with_keyring_without_external_binary(monkeypatch) -> None:
    class _SecretServiceBackend:
        __module__ = "keyring.backends.secretservice"

    class _FakeKeyringModule:
        def get_keyring(self):
            return _SecretServiceBackend()

        def get_password(self, _service: str, _account: str):
            return None

        def set_password(self, _service: str, _account: str, _value: str):
            return None

    monkeypatch.setattr("dnadesign.notify.secrets._load_keyring_module", lambda: _FakeKeyringModule())
    monkeypatch.setattr("dnadesign.notify.secrets.shutil.which", lambda _cmd: None)
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)

    assert is_secret_backend_available("secretservice") is True


def test_secretservice_keyring_probe_failure_falls_back_to_command(monkeypatch) -> None:
    class _SecretServiceBackend:
        __module__ = "keyring.backends.secretservice"

    class _FakeKeyringModule:
        def get_keyring(self):
            return _SecretServiceBackend()

        def get_password(self, _service: str, _account: str):
            raise RuntimeError("keyring unavailable")

        def set_password(self, _service: str, _account: str, _value: str):
            return None

    monkeypatch.setattr("dnadesign.notify.secrets._load_keyring_module", lambda: _FakeKeyringModule())
    monkeypatch.setattr("dnadesign.notify.secrets.shutil.which", lambda _cmd: "/usr/bin/secret-tool")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/tmp/dbus-test")
    monkeypatch.setattr(
        "dnadesign.notify.secrets.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=""),
    )

    assert is_secret_backend_available("secretservice") is True


def test_secretservice_store_and_resolve_use_keyring_without_external_binary(monkeypatch) -> None:
    class _SecretServiceBackend:
        __module__ = "keyring.backends.secretservice"

    class _FakeKeyringModule:
        def __init__(self):
            self.values: dict[tuple[str, str], str] = {}

        def get_keyring(self):
            return _SecretServiceBackend()

        def get_password(self, service: str, account: str):
            return self.values.get((service, account))

        def set_password(self, service: str, account: str, value: str):
            self.values[(service, account)] = value

    fake_keyring = _FakeKeyringModule()
    monkeypatch.setattr("dnadesign.notify.secrets._load_keyring_module", lambda: fake_keyring)
    monkeypatch.setattr("dnadesign.notify.secrets.shutil.which", lambda _cmd: None)
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)

    secret_ref = "secretservice://dnadesign.notify/default"
    store_secret_ref(secret_ref, "https://example.invalid/webhook")
    assert resolve_secret_ref(secret_ref) == "https://example.invalid/webhook"
