"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/delivery/secrets/contract.py

Secret reference contracts and parsing for notify secret backends.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from ...errors import NotifyConfigError

BACKEND_KEYCHAIN = "keychain"
BACKEND_SECRET_SERVICE = "secretservice"  # pragma: allowlist secret
BACKEND_FILE = "file"
SUPPORTED_SECRET_BACKENDS = {BACKEND_KEYCHAIN, BACKEND_SECRET_SERVICE, BACKEND_FILE}


@dataclass(frozen=True)
class SecretReference:
    backend: str
    service: str | None = None
    account: str | None = None
    file_path: Path | None = None


def normalize_backend(value: str) -> str:
    backend = str(value or "").strip().lower()
    if backend not in SUPPORTED_SECRET_BACKENDS:
        allowed = ", ".join(sorted(SUPPORTED_SECRET_BACKENDS))
        raise NotifyConfigError(f"unsupported secret backend '{value}'; expected one of: {allowed}")
    return backend


def parse_secret_ref(secret_ref: str) -> SecretReference:
    value = str(secret_ref or "").strip()
    if not value:
        raise NotifyConfigError("secret_ref must be a non-empty string")
    parsed = urlparse(value)
    backend = normalize_backend(parsed.scheme)
    if backend == BACKEND_FILE:
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


def backend_command(backend: str) -> str:
    normalized = normalize_backend(backend)
    if normalized == BACKEND_KEYCHAIN:
        return "security"
    if normalized == BACKEND_FILE:
        return "filesystem"
    return "secret-tool"
