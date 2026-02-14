"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/validation.py

Validation helpers for notifier inputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from .errors import NotifyConfigError
from .secrets import resolve_secret_ref

_SLACK_WEBHOOK_HOSTS = {"hooks.slack.com", "hooks.slack-gov.com"}


def resolve_webhook_url(*, url: str | None, url_env: str | None, secret_ref: str | None = None) -> str:
    source_count = int(bool(url)) + int(bool(url_env)) + int(bool(secret_ref))
    if source_count != 1:
        raise NotifyConfigError("Specify exactly one of --url, --url-env, or --secret-ref.")
    if url_env:
        env_value = os.environ.get(url_env, "").strip()
        if not env_value:
            raise NotifyConfigError(f"--url-env {url_env} is not set or empty.")
        resolved = env_value
    elif secret_ref:
        resolved = resolve_secret_ref(secret_ref)
    else:
        resolved = str(url).strip() if url is not None else ""
    if not resolved:
        raise NotifyConfigError("Webhook URL is empty.")
    parsed = urlparse(resolved)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise NotifyConfigError("Webhook URL must be http(s) with a host.")
    return resolved


def resolve_tls_ca_bundle(*, webhook_url: str, tls_ca_bundle: Path | None) -> Path | None:
    parsed = urlparse(str(webhook_url).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise NotifyConfigError("Webhook URL must be http(s) with a host.")
    if parsed.scheme == "http":
        if tls_ca_bundle is None:
            return None
        resolved_http = tls_ca_bundle.expanduser().resolve()
        if not resolved_http.exists():
            raise NotifyConfigError(f"CA bundle file not found: {resolved_http}")
        if not resolved_http.is_file():
            raise NotifyConfigError(f"CA bundle path is not a file: {resolved_http}")
        return resolved_http

    candidate = tls_ca_bundle
    if candidate is None:
        env_value = os.environ.get("SSL_CERT_FILE", "").strip()
        if env_value:
            candidate = Path(env_value)
    if candidate is None:
        raise NotifyConfigError(
            "HTTPS webhook delivery requires an explicit CA bundle. "
            "Pass --tls-ca-bundle or set SSL_CERT_FILE."
        )

    resolved = candidate.expanduser().resolve()
    if not resolved.exists():
        raise NotifyConfigError(f"CA bundle file not found: {resolved}")
    if not resolved.is_file():
        raise NotifyConfigError(f"CA bundle path is not a file: {resolved}")
    return resolved


def validate_provider_webhook_url(*, provider: str, webhook_url: str) -> None:
    provider_name = str(provider or "").strip().lower()
    parsed = urlparse(str(webhook_url).strip())
    host = str(parsed.hostname or "").strip().lower()
    if provider_name != "slack":
        return
    if host not in _SLACK_WEBHOOK_HOSTS:
        allowed = ", ".join(sorted(_SLACK_WEBHOOK_HOSTS))
        raise NotifyConfigError(
            f"slack provider requires webhook host in {{{allowed}}}; found: {host or '<missing>'}"
        )
    if not str(parsed.path or "").startswith("/services/"):
        raise NotifyConfigError("slack provider requires webhook path beginning with /services/")
