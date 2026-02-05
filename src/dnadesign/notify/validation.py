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
from urllib.parse import urlparse

from .errors import NotifyConfigError


def resolve_webhook_url(*, url: str | None, url_env: str | None) -> str:
    if bool(url) == bool(url_env):
        raise NotifyConfigError("Specify exactly one of --url or --url-env.")
    if url_env:
        env_value = os.environ.get(url_env, "").strip()
        if not env_value:
            raise NotifyConfigError(f"--url-env {url_env} is not set or empty.")
        resolved = env_value
    else:
        resolved = str(url).strip() if url is not None else ""
    if not resolved:
        raise NotifyConfigError("Webhook URL is empty.")
    parsed = urlparse(resolved)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise NotifyConfigError("Webhook URL must be http(s) with a host.")
    return resolved
