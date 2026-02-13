"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/secrets.py

Secret reference parsing and OS-backed secret storage helpers for webhook URLs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from types import ModuleType
from urllib.parse import urlparse

from .errors import NotifyConfigError

_BACKEND_KEYCHAIN = "keychain"
_BACKEND_SECRET_SERVICE = "secretservice"  # pragma: allowlist secret
_SUPPORTED_SECRET_BACKENDS = {_BACKEND_KEYCHAIN, _BACKEND_SECRET_SERVICE}


@dataclass(frozen=True)
class SecretReference:
    backend: str
    service: str
    account: str


def _normalize_backend(value: str) -> str:
    backend = str(value or "").strip().lower()
    if backend not in _SUPPORTED_SECRET_BACKENDS:
        allowed = ", ".join(sorted(_SUPPORTED_SECRET_BACKENDS))
        raise NotifyConfigError(f"unsupported secret backend '{value}'; expected one of: {allowed}")
    return backend


def parse_secret_ref(secret_ref: str) -> SecretReference:
    value = str(secret_ref or "").strip()
    if not value:
        raise NotifyConfigError("secret_ref must be a non-empty string")
    parsed = urlparse(value)
    backend = _normalize_backend(parsed.scheme)
    service = str(parsed.netloc or "").strip()
    account = str(parsed.path or "").strip("/ ")
    if not service or not account:
        raise NotifyConfigError(
            "secret_ref must look like keychain://<service>/<account> or secretservice://<service>/<account>"
        )
    return SecretReference(backend=backend, service=service, account=account)


def _backend_command(backend: str) -> str:
    normalized = _normalize_backend(backend)
    if normalized == _BACKEND_KEYCHAIN:
        return "security"
    return "secret-tool"


def _load_keyring_module() -> ModuleType | None:
    try:
        return importlib.import_module("keyring")
    except Exception:
        return None


def _keyring_backend_descriptor(keyring_module: ModuleType) -> str:
    try:
        backend = keyring_module.get_keyring()
    except Exception:
        return ""
    return f"{type(backend).__module__}.{type(backend).__name__}".lower()


def _keyring_backend_matches(*, backend: str, descriptor: str) -> bool:
    descriptor_value = str(descriptor or "").strip().lower()
    if not descriptor_value:
        return False
    if backend == _BACKEND_KEYCHAIN:
        return "keychain" in descriptor_value or "macos" in descriptor_value
    return "secretservice" in descriptor_value


def _keyring_client_for_backend(backend: str) -> ModuleType | None:
    keyring_module = _load_keyring_module()
    if keyring_module is None:
        return None
    descriptor = _keyring_backend_descriptor(keyring_module)
    if not _keyring_backend_matches(backend=backend, descriptor=descriptor):
        return None
    if not hasattr(keyring_module, "get_password") or not hasattr(keyring_module, "set_password"):
        return None
    return keyring_module


def _probe_command(*, args: list[str], ok_codes: set[int], timeout_seconds: float = 2.0) -> bool:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return int(result.returncode) in ok_codes


def is_secret_backend_available(backend: str) -> bool:
    normalized = _normalize_backend(backend)
    keyring_module = _keyring_client_for_backend(normalized)
    if keyring_module is not None:
        try:
            keyring_module.get_password("dnadesign.notify.probe", "dnadesign.notify.probe")
            return True
        except Exception:
            pass

    command = _backend_command(normalized)
    if shutil.which(command) is None:
        return False

    if normalized == _BACKEND_KEYCHAIN:
        return _probe_command(
            args=[
                "security",
                "find-generic-password",
                "-s",
                "dnadesign.notify.probe",
                "-a",
                "dnadesign.notify.probe",
                "-w",
            ],
            ok_codes={0, 44},
        )

    dbus_address = str(os.environ.get("DBUS_SESSION_BUS_ADDRESS", "")).strip()
    if not dbus_address:
        return False
    return _probe_command(
        args=[
            "secret-tool",
            "lookup",
            "service",
            "dnadesign.notify.probe",
            "account",
            "dnadesign.notify.probe",
        ],
        ok_codes={0, 1},
    )


def _run_command(args: list[str], *, input_text: str | None = None) -> str:
    try:
        result = subprocess.run(
            args,
            input=input_text,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise NotifyConfigError(f"failed to run secret backend command '{args[0]}'") from exc
    if result.returncode != 0:
        message = str(result.stderr or result.stdout or "").strip() or "unknown command failure"
        raise NotifyConfigError(f"secret backend command failed: {message}")
    return str(result.stdout or "").strip()


def resolve_secret_ref(secret_ref: str) -> str:
    parsed = parse_secret_ref(secret_ref)
    keyring_module = _keyring_client_for_backend(parsed.backend)
    if keyring_module is not None:
        try:
            value = keyring_module.get_password(parsed.service, parsed.account)
        except Exception as exc:
            raise NotifyConfigError("secret backend keyring read failed") from exc
        if not isinstance(value, str) or not value.strip():
            raise NotifyConfigError(f"secret value not found for reference: {secret_ref}")
        return value

    if not is_secret_backend_available(parsed.backend):
        command = _backend_command(parsed.backend)
        raise NotifyConfigError(f"secret backend '{parsed.backend}' is not available; missing command: {command}")

    if parsed.backend == _BACKEND_KEYCHAIN:
        return _run_command(
            [
                "security",
                "find-generic-password",
                "-s",
                parsed.service,
                "-a",
                parsed.account,
                "-w",
            ]
        )

    return _run_command(["secret-tool", "lookup", "service", parsed.service, "account", parsed.account])


def store_secret_ref(secret_ref: str, secret_value: str) -> None:
    parsed = parse_secret_ref(secret_ref)
    value = str(secret_value or "").strip()
    if not value:
        raise NotifyConfigError("webhook URL value must be non-empty when storing in a secret backend")
    keyring_module = _keyring_client_for_backend(parsed.backend)
    if keyring_module is not None:
        try:
            keyring_module.set_password(parsed.service, parsed.account, value)
        except Exception as exc:
            raise NotifyConfigError("secret backend keyring write failed") from exc
        return

    if not is_secret_backend_available(parsed.backend):
        command = _backend_command(parsed.backend)
        raise NotifyConfigError(f"secret backend '{parsed.backend}' is not available; missing command: {command}")

    if parsed.backend == _BACKEND_KEYCHAIN:
        _run_command(
            [
                "security",
                "add-generic-password",
                "-U",
                "-s",
                parsed.service,
                "-a",
                parsed.account,
                "-w",
                value,
            ]
        )
        return

    _run_command(
        [
            "secret-tool",
            "store",
            "--label",
            f"dnadesign notify webhook ({parsed.service}/{parsed.account})",
            "service",
            parsed.service,
            "account",
            parsed.account,
        ],
        input_text=f"{value}\n",
    )
