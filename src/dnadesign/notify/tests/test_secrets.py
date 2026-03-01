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
from pathlib import Path

import pytest

from dnadesign.notify.delivery.secrets import (
    is_secret_backend_available,
    parse_secret_ref,
    resolve_secret_ref,
    store_secret_ref,
)
from dnadesign.notify.errors import NotifyConfigError


def test_secretservice_unavailable_without_dbus_session(monkeypatch) -> None:
    monkeypatch.setattr("dnadesign.notify.delivery.secrets.ops.shutil.which", lambda _cmd: "/usr/bin/secret-tool")
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)

    assert is_secret_backend_available("secretservice") is False


def test_secretservice_available_with_probeable_runtime(monkeypatch) -> None:
    monkeypatch.setattr("dnadesign.notify.delivery.secrets.ops.shutil.which", lambda _cmd: "/usr/bin/secret-tool")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/tmp/dbus-test")
    monkeypatch.setattr(
        "dnadesign.notify.delivery.secrets.shell_backend.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=""),
    )

    assert is_secret_backend_available("secretservice") is True


def test_keychain_availability_uses_runtime_probe(monkeypatch) -> None:
    monkeypatch.setattr("dnadesign.notify.delivery.secrets.ops.shutil.which", lambda _cmd: "/usr/bin/security")
    monkeypatch.setattr(
        "dnadesign.notify.delivery.secrets.shell_backend.subprocess.run",
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

    monkeypatch.setattr(
        "dnadesign.notify.delivery.secrets.keyring_backend.load_keyring_module",
        lambda: _FakeKeyringModule(),
    )
    monkeypatch.setattr("dnadesign.notify.delivery.secrets.ops.shutil.which", lambda _cmd: None)
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

    monkeypatch.setattr(
        "dnadesign.notify.delivery.secrets.keyring_backend.load_keyring_module",
        lambda: _FakeKeyringModule(),
    )
    monkeypatch.setattr("dnadesign.notify.delivery.secrets.ops.shutil.which", lambda _cmd: "/usr/bin/secret-tool")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/tmp/dbus-test")
    monkeypatch.setattr(
        "dnadesign.notify.delivery.secrets.shell_backend.subprocess.run",
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
    monkeypatch.setattr("dnadesign.notify.delivery.secrets.keyring_backend.load_keyring_module", lambda: fake_keyring)
    monkeypatch.setattr("dnadesign.notify.delivery.secrets.ops.shutil.which", lambda _cmd: None)
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)

    secret_ref = "secretservice://dnadesign.notify/default"  # pragma: allowlist secret
    store_secret_ref(secret_ref, "https://example.invalid/webhook")
    assert resolve_secret_ref(secret_ref) == "https://example.invalid/webhook"


def test_file_secret_store_and_resolve_round_trip(tmp_path: Path) -> None:
    secret_path = (tmp_path / "notify" / "secret.txt").resolve()
    secret_ref = secret_path.as_uri()
    store_secret_ref(secret_ref, "https://example.invalid/webhook")
    assert resolve_secret_ref(secret_ref) == "https://example.invalid/webhook"


def test_parse_file_secret_ref_requires_absolute_path() -> None:
    with pytest.raises(NotifyConfigError, match="must not include a host"):
        parse_secret_ref("file://relative/path")
