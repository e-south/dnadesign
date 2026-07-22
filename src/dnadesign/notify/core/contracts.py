"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/core/contracts.py

Public notify contracts for webhook profile parsing and TLS CA bundle.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Collection, Literal, Mapping, Sequence
from urllib.parse import unquote, urlparse

DEFAULT_NOTIFY_WEBHOOK_SOURCES = frozenset({"env", "secret_ref"})
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


def parse_notify_profile_webhook(
    profile_data: Mapping[str, Any],
    *,
    required_profile_version: int | None = None,
    allowed_sources: Collection[str] = DEFAULT_NOTIFY_WEBHOOK_SOURCES,
) -> tuple[str, str]:
    if required_profile_version is not None:
        version = profile_data.get("profile_version")
        if version != required_profile_version:
            raise ValueError(f"profile_version must be {required_profile_version}; found {version!r}")
    webhook = profile_data.get("webhook")
    if not isinstance(webhook, Mapping):
        raise ValueError("profile field 'webhook' must be an object")
    source = str(webhook.get("source") or "").strip().lower()
    ref = str(webhook.get("ref") or "").strip()
    normalized_allowed = tuple(sorted({str(value).strip().lower() for value in allowed_sources if str(value).strip()}))
    if source not in normalized_allowed:
        allowed = ", ".join(normalized_allowed)
        raise ValueError(f"profile field 'webhook.source' must be one of: {allowed}")
    if not ref:
        raise ValueError("profile field 'webhook.ref' must be a non-empty string")
    return source, ref


def resolve_file_secret_ref_path(secret_ref: str, *, source_label: str) -> Path:
    parsed = urlparse(secret_ref)
    if parsed.scheme != "file":
        raise ValueError(f"{source_label} must use file:// URI: {secret_ref}")
    if parsed.netloc:
        raise ValueError(f"{source_label} must not include host for file:// references: {secret_ref}")
    if not parsed.path:
        raise ValueError(f"{source_label} path is missing: {secret_ref}")
    return Path(unquote(parsed.path)).expanduser().resolve()


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


__all__ = [
    "DEFAULT_NOTIFY_WEBHOOK_SOURCES",
    "DEFAULT_SYSTEM_TLS_CA_BUNDLE_CANDIDATES",
    "TLSCABundleResolutionError",
    "parse_notify_profile_webhook",
    "resolve_file_secret_ref_path",
    "resolve_tls_ca_bundle_path",
]
