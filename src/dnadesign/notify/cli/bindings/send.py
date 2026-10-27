"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/bindings/send.py

Send command binding implementation for notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_send_impl(
    *,
    deps: Any,
    status: str,
    tool: str,
    run_id: str,
    provider: str,
    url: str | None,
    url_env: str | None,
    secret_ref: str | None,
    tls_ca_bundle: Path | None,
    message: str | None,
    meta: Path | None,
    timeout: float,
    retries: int,
    dry_run: bool,
) -> None:
    deps.run_send_command(
        status=status,
        tool=tool,
        run_id=run_id,
        provider=provider,
        url=url,
        url_env=url_env,
        secret_ref=secret_ref,
        tls_ca_bundle=tls_ca_bundle,
        message=message,
        meta=meta,
        timeout=timeout,
        retries=retries,
        dry_run=dry_run,
        load_meta_fn=deps._load_meta,
        resolve_webhook_url_fn=deps.resolve_webhook_url,
        validate_provider_webhook_url_fn=deps.validate_provider_webhook_url,
        build_payload_fn=deps.build_payload,
        format_for_provider_fn=deps.format_for_provider,
        resolve_tls_ca_bundle_fn=deps.resolve_tls_ca_bundle,
        post_json_fn=lambda webhook_url, payload, timeout, retries, tls_ca_bundle: deps.post_json(
            webhook_url,
            payload,
            timeout=timeout,
            retries=retries,
            tls_ca_bundle=tls_ca_bundle,
        ),
    )
