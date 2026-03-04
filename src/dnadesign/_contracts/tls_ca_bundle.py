"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/_contracts/tls_ca_bundle.py

Shared fail-fast TLS CA bundle resolution contract for HTTPS delivery paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Sequence

DEFAULT_SYSTEM_TLS_CA_BUNDLE_CANDIDATES = (
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/certs/ca-certificates.crt",
)

SourceKind = Literal["explicit", "env", "none"]
ReasonKind = Literal["missing", "not_file", "unreadable", "not_configured"]


class TLSCABundleResolutionError(ValueError):
    def __init__(
        self,
        *,
        reason: ReasonKind,
        source: SourceKind,
        path: Path | None = None,
        env_var_name: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.source = source
        self.path = path
        self.env_var_name = env_var_name


def _validate_file_path(path: Path, *, source: SourceKind, env_var_name: str | None) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise TLSCABundleResolutionError(
            reason="missing",
            source=source,
            path=resolved,
            env_var_name=env_var_name,
        )
    if not resolved.is_file():
        raise TLSCABundleResolutionError(
            reason="not_file",
            source=source,
            path=resolved,
            env_var_name=env_var_name,
        )
    if not os.access(resolved, os.R_OK):
        raise TLSCABundleResolutionError(
            reason="unreadable",
            source=source,
            path=resolved,
            env_var_name=env_var_name,
        )
    return resolved


def resolve_tls_ca_bundle_path(
    *,
    explicit_path: Path | None,
    env_var_name: str = "SSL_CERT_FILE",
    allow_system_candidates: bool,
    system_candidates: Sequence[str] = DEFAULT_SYSTEM_TLS_CA_BUNDLE_CANDIDATES,
    not_configured_error: str,
    source_label: str,
) -> Path:
    if explicit_path is not None:
        try:
            return _validate_file_path(explicit_path, source="explicit", env_var_name=None)
        except TLSCABundleResolutionError as exc:
            assert exc.path is not None
            if exc.reason in {"missing", "not_file"}:
                raise ValueError(f"{source_label} does not exist or is not a file: {exc.path}") from exc
            if exc.reason == "unreadable":
                raise ValueError(f"{source_label} is not readable: {exc.path}") from exc
            raise

    env_value = os.environ.get(env_var_name, "").strip()
    if env_value:
        try:
            return _validate_file_path(Path(env_value), source="env", env_var_name=env_var_name)
        except TLSCABundleResolutionError as exc:
            assert exc.path is not None
            if exc.reason in {"missing", "not_file"}:
                raise ValueError(
                    f"{source_label} from {env_var_name} does not exist or is not a file: {exc.path}"
                ) from exc
            if exc.reason == "unreadable":
                raise ValueError(f"{source_label} from {env_var_name} is not readable: {exc.path}") from exc
            raise

    if allow_system_candidates:
        for candidate in system_candidates:
            path = Path(str(candidate).strip()).expanduser()
            if not str(path):
                continue
            try:
                return _validate_file_path(path, source="explicit", env_var_name=None)
            except TLSCABundleResolutionError:
                continue

    raise ValueError(not_configured_error)
