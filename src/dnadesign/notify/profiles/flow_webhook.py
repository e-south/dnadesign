"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/profiles/flow_webhook.py

Webhook source resolution and secure secret storage helpers for notify setup.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer

from ..delivery.secrets import is_secret_backend_available, resolve_secret_ref, store_secret_ref
from ..errors import NotifyConfigError
from .policy import DEFAULT_WEBHOOK_ENV
from .resolve import resolve_cli_optional_string


def _default_file_secret_path(secret_name: str) -> Path:
    notify_root = Path(__file__).resolve().parents[1]
    return (notify_root / ".secrets" / f"{secret_name}.webhook").resolve()


def resolve_webhook_config(
    *,
    secret_source: str,
    url_env: str | None,
    secret_ref: str | None,
    webhook_url: str | None,
    store_webhook: bool,
    secret_name: str,
    secret_backend_available_fn: Callable[[str], bool] = is_secret_backend_available,
    resolve_secret_ref_fn: Callable[[str], str] = resolve_secret_ref,
    store_secret_ref_fn: Callable[[str, str], None] = store_secret_ref,
) -> dict[str, str]:
    mode = str(secret_source or "").strip().lower()
    if not mode:
        raise NotifyConfigError("secret_source must be a non-empty string")
    if mode not in {"auto", "env", "keychain", "secretservice", "file"}:
        raise NotifyConfigError("secret_source must be one of: auto, env, keychain, secretservice, file")

    if mode == "env":
        env_name = resolve_cli_optional_string(field="url_env", cli_value=url_env)
        if env_name is None:
            env_name = DEFAULT_WEBHOOK_ENV
        return {"source": "env", "ref": env_name}

    provided_secret_ref = resolve_cli_optional_string(field="secret_ref", cli_value=secret_ref)
    secret_refs: list[str]
    if provided_secret_ref is not None:
        secret_refs = [provided_secret_ref]
    elif mode == "auto":
        candidate_modes: list[str] = []
        for candidate in ("keychain", "secretservice", "file"):
            if secret_backend_available_fn(candidate):
                candidate_modes.append(candidate)
        if not candidate_modes:
            raise NotifyConfigError(
                "secret_source=auto requires keychain, secretservice, or file backend availability. "
                "Pass --secret-source env to opt into environment-variable webhook storage."
            )
        secret_refs = []
        for candidate in candidate_modes:
            if candidate == "file":
                secret_path = _default_file_secret_path(secret_name)
                secret_refs.append(secret_path.as_uri())
            else:
                secret_refs.append(f"{candidate}://dnadesign.notify/{secret_name}")
    else:
        if not secret_backend_available_fn(mode):
            raise NotifyConfigError(f"secret backend '{mode}' is not available on this system")
        if mode == "file":
            secret_path = _default_file_secret_path(secret_name)
            secret_refs = [secret_path.as_uri()]
        else:
            secret_refs = [f"{mode}://dnadesign.notify/{secret_name}"]

    if not store_webhook:
        return {"source": "secret_ref", "ref": secret_refs[0]}

    webhook_value = resolve_cli_optional_string(field="webhook_url", cli_value=webhook_url)
    last_error: NotifyConfigError | None = None
    for secret_value in secret_refs:
        webhook_config = {"source": "secret_ref", "ref": secret_value}
        if webhook_value is None:
            try:
                _ = resolve_secret_ref_fn(secret_value)
                return webhook_config
            except NotifyConfigError:
                webhook_value = str(typer.prompt("Webhook URL", hide_input=True)).strip()
        if not webhook_value:
            raise NotifyConfigError("webhook_url is required when --store-webhook is enabled")
        try:
            store_secret_ref_fn(secret_value, webhook_value)
            return webhook_config
        except NotifyConfigError as exc:
            last_error = exc
            if len(secret_refs) == 1:
                raise
            continue

    if last_error is not None:
        raise last_error
    raise NotifyConfigError("failed to resolve webhook secret configuration")
