"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/_contracts/notify_webhook_profile.py

Shared webhook source/ref parsing helpers for notify profile payloads.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Mapping
from urllib.parse import unquote, urlparse

DEFAULT_NOTIFY_WEBHOOK_SOURCES = frozenset({"env", "secret_ref"})


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
