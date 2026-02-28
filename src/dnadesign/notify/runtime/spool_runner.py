"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/runtime/spool_runner.py

Runtime orchestration for notify spool drain execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..errors import NotifyConfigError, NotifyDeliveryError, NotifyError


def run_spool_drain(
    *,
    spool_dir: Path | None,
    provider: str | None,
    url: str | None,
    url_env: str | None,
    secret_ref: str | None,
    tls_ca_bundle: Path | None,
    profile: Path | None,
    connect_timeout: float,
    read_timeout: float,
    retry_max: int,
    retry_base_seconds: float,
    fail_fast: bool,
    read_profile: Callable[[Path | None], dict[str, Any]],
    resolve_cli_optional_string: Callable[..., str | None],
    resolve_path_value: Callable[..., Path],
    resolve_profile_webhook_source: Callable[[dict[str, Any]], tuple[str | None, str | None]],
    resolve_optional_path_value: Callable[..., Path | None],
    resolve_webhook_url: Callable[..., str],
    resolve_tls_ca_bundle: Callable[..., Path | None],
    validate_provider_webhook_url: Callable[..., None],
    format_for_provider: Callable[[str, dict[str, Any]], dict[str, Any]],
    post_with_backoff: Callable[..., None],
) -> None:
    profile_path = profile.expanduser().resolve() if profile is not None else None
    profile_data = read_profile(profile_path) if profile_path is not None else {}
    provider_override = resolve_cli_optional_string(field="provider", cli_value=provider)
    profile_provider_value = profile_data.get("provider")
    profile_provider: str | None = None
    if profile_provider_value is not None:
        if not isinstance(profile_provider_value, str) or not profile_provider_value.strip():
            raise NotifyConfigError("profile field 'provider' must be a non-empty string")
        profile_provider = profile_provider_value.strip()
    spool_dir_value = resolve_path_value(
        field="spool_dir",
        cli_value=spool_dir,
        profile_data=profile_data,
        profile_path=profile_path,
    )
    profile_url_env, profile_secret_ref = resolve_profile_webhook_source(profile_data)
    url_env_value = resolve_cli_optional_string(field="url_env", cli_value=url_env)
    if url_env_value is None:
        url_env_value = profile_url_env
    secret_ref_value = resolve_cli_optional_string(field="secret_ref", cli_value=secret_ref)
    if secret_ref_value is None:
        secret_ref_value = profile_secret_ref
    profile_tls_ca_bundle = resolve_optional_path_value(
        field="tls_ca_bundle",
        cli_value=tls_ca_bundle,
        profile_data=profile_data,
        profile_path=profile_path,
    )
    webhook_url = resolve_webhook_url(url=url, url_env=url_env_value, secret_ref=secret_ref_value)
    resolved_tls_ca_bundle = resolve_tls_ca_bundle(
        webhook_url=webhook_url,
        tls_ca_bundle=profile_tls_ca_bundle,
    )
    if not spool_dir_value.exists():
        raise NotifyConfigError(f"spool directory not found: {spool_dir_value}")
    failed = 0
    for path in sorted(spool_dir_value.glob("*.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
            payload = body.get("payload")
            if not isinstance(payload, dict):
                raise NotifyConfigError(f"invalid spool payload file: {path}")
            spool_provider_value = body.get("provider")
            spool_provider: str | None = None
            if spool_provider_value is not None:
                if not isinstance(spool_provider_value, str) or not spool_provider_value.strip():
                    raise NotifyConfigError(f"invalid spool payload provider value: {path}")
                spool_provider = spool_provider_value.strip()
            provider_value = provider_override or spool_provider or profile_provider
            if not provider_value:
                raise NotifyConfigError(
                    f"spool payload missing provider: {path}. Pass --provider or include provider in spool files."
                )
            validate_provider_webhook_url(provider=provider_value, webhook_url=webhook_url)
            formatted = format_for_provider(provider_value, payload)
            post_with_backoff(
                webhook_url,
                formatted,
                tls_ca_bundle=resolved_tls_ca_bundle,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                retry_max=retry_max,
                retry_base_seconds=retry_base_seconds,
            )
            path.unlink()
        except NotifyError:
            failed += 1
            if fail_fast:
                raise
    if failed:
        raise NotifyDeliveryError(f"{failed} spool file(s) failed delivery")
