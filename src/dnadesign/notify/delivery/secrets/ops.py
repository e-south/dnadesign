"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/delivery/secrets/ops.py

Runtime availability, resolve, and store operations for notify secrets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil

from ...errors import NotifyConfigError
from .contract import BACKEND_FILE, BACKEND_KEYCHAIN, backend_command, normalize_backend, parse_secret_ref
from .file_backend import read_file_secret, write_file_secret
from .keyring_backend import keyring_client_for_backend
from .shell_backend import probe_command, run_command


def is_secret_backend_available(backend: str) -> bool:
    normalized = normalize_backend(backend)
    if normalized == BACKEND_FILE:
        return True
    keyring_module = keyring_client_for_backend(normalized)
    if keyring_module is not None:
        try:
            keyring_module.get_password("dnadesign.notify.probe", "dnadesign.notify.probe")
            return True
        except Exception:
            pass

    command = backend_command(normalized)
    if shutil.which(command) is None:
        return False

    if normalized == BACKEND_KEYCHAIN:
        return probe_command(
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
    return probe_command(
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


def resolve_secret_ref(secret_ref: str) -> str:
    parsed = parse_secret_ref(secret_ref)
    if parsed.backend == BACKEND_FILE:
        if parsed.file_path is None:
            raise NotifyConfigError("file secret_ref must include an absolute path")
        return read_file_secret(parsed.file_path)
    keyring_module = keyring_client_for_backend(parsed.backend)
    if keyring_module is not None:
        try:
            value = keyring_module.get_password(parsed.service, parsed.account)
        except Exception as exc:
            raise NotifyConfigError("secret backend keyring read failed") from exc
        if not isinstance(value, str) or not value.strip():
            raise NotifyConfigError(f"secret value not found for reference: {secret_ref}")
        return value

    if not is_secret_backend_available(parsed.backend):
        command = backend_command(parsed.backend)
        raise NotifyConfigError(f"secret backend '{parsed.backend}' is not available; missing command: {command}")

    if parsed.backend == BACKEND_KEYCHAIN:
        return run_command(
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

    return run_command(["secret-tool", "lookup", "service", parsed.service, "account", parsed.account])


def store_secret_ref(secret_ref: str, secret_value: str) -> None:
    parsed = parse_secret_ref(secret_ref)
    value = str(secret_value or "").strip()
    if not value:
        raise NotifyConfigError("webhook URL value must be non-empty when storing in a secret backend")
    if parsed.backend == BACKEND_FILE:
        if parsed.file_path is None:
            raise NotifyConfigError("file secret_ref must include an absolute path")
        write_file_secret(parsed.file_path, value)
        return
    keyring_module = keyring_client_for_backend(parsed.backend)
    if keyring_module is not None:
        try:
            keyring_module.set_password(parsed.service, parsed.account, value)
        except Exception as exc:
            raise NotifyConfigError("secret backend keyring write failed") from exc
        return

    if not is_secret_backend_available(parsed.backend):
        command = backend_command(parsed.backend)
        raise NotifyConfigError(f"secret backend '{parsed.backend}' is not available; missing command: {command}")

    if parsed.backend == BACKEND_KEYCHAIN:
        run_command(
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

    run_command(
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
