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
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from urllib.parse import urlparse

from .errors import NotifyConfigError

_BACKEND_KEYCHAIN = "keychain"
_BACKEND_SECRET_SERVICE = "secretservice"  # pragma: allowlist secret
_BACKEND_FILE = "file"
_SUPPORTED_SECRET_BACKENDS = {_BACKEND_KEYCHAIN, _BACKEND_SECRET_SERVICE, _BACKEND_FILE}


@dataclass(frozen=True)
class SecretReference:
    backend: str
    service: str | None = None
    account: str | None = None
    file_path: Path | None = None


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
    if backend == _BACKEND_FILE:
        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            raise NotifyConfigError("file secret_ref must not include a host; expected file:///absolute/path/to/secret")
        path_value = str(parsed.path or "").strip()
        if not path_value:
            raise NotifyConfigError("file secret_ref must include an absolute path")
        file_path = Path(path_value).expanduser()
        if not file_path.is_absolute():
            raise NotifyConfigError("file secret_ref must include an absolute path")
        return SecretReference(backend=backend, file_path=file_path.resolve())
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
    if normalized == _BACKEND_FILE:
        return "filesystem"
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
    if normalized == _BACKEND_FILE:
        return True
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


def _ensure_private_secret_parent(path: Path) -> None:
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise NotifyConfigError(f"failed to create secret directory: {parent}") from exc
    if os.name == "nt":
        return
    try:
        parent.chmod(0o700)
    except OSError as exc:
        raise NotifyConfigError(f"failed to set secure permissions on secret directory: {parent}") from exc
    mode = stat.S_IMODE(parent.stat().st_mode)
    if mode & 0o077:
        raise NotifyConfigError(
            f"secret directory must not be group/world-accessible (expected mode 700): {parent} (mode={oct(mode)})"
        )


def _assert_private_secret_file(path: Path) -> None:
    if not path.exists():
        raise NotifyConfigError(f"secret value not found for reference: {path.as_uri()}")
    if not path.is_file():
        raise NotifyConfigError(f"secret path is not a file: {path}")
    if os.name == "nt":
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise NotifyConfigError(
            f"secret file must not be group/world-accessible (expected mode 600): {path} (mode={oct(mode)})"
        )


def _read_file_secret(path: Path) -> str:
    _assert_private_secret_file(path)
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise NotifyConfigError(f"failed to read secret file: {path}") from exc
    if not value:
        raise NotifyConfigError(f"secret value not found for reference: {path.as_uri()}")
    return value


def _write_file_secret(path: Path, value: str) -> None:
    _ensure_private_secret_parent(path)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        tmp_path.write_text(value, encoding="utf-8")
        if os.name != "nt":
            tmp_path.chmod(0o600)
        tmp_path.replace(path)
        if os.name != "nt":
            path.chmod(0o600)
    except OSError as exc:
        raise NotifyConfigError(f"failed to write secret file: {path}") from exc
    _assert_private_secret_file(path)


def resolve_secret_ref(secret_ref: str) -> str:
    parsed = parse_secret_ref(secret_ref)
    if parsed.backend == _BACKEND_FILE:
        if parsed.file_path is None:
            raise NotifyConfigError("file secret_ref must include an absolute path")
        return _read_file_secret(parsed.file_path)
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
    if parsed.backend == _BACKEND_FILE:
        if parsed.file_path is None:
            raise NotifyConfigError("file secret_ref must include an absolute path")
        _write_file_secret(parsed.file_path, value)
        return
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
