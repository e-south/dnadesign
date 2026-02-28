"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/runtime/watch_delivery.py

Delivery and dry-run emission operations for notify watch runtime.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from ..errors import NotifyConfigError, NotifyDeliveryError
from .spool import spool_payload as _spool_payload


@dataclass(frozen=True)
class DeliveryOutcome:
    cursor_advanced: bool
    terminal_reached: bool
    failed_unsent_delta: int


def emit_or_deliver_event(
    *,
    status_value: str,
    payload: dict[str, Any],
    formatted_payload: dict[str, Any],
    dry_run: bool,
    stop_on_terminal_status: bool,
    webhook_url: str | None,
    resolved_tls_ca_bundle: Path | None,
    connect_timeout: float,
    read_timeout: float,
    retry_max: int,
    retry_base_seconds: float,
    fail_fast: bool,
    spool_dir_value: Path | None,
    provider_value: str,
    post_with_backoff: Callable[..., None],
) -> DeliveryOutcome:
    if dry_run:
        typer.echo(json.dumps(formatted_payload, sort_keys=True))
        return DeliveryOutcome(
            cursor_advanced=True,
            terminal_reached=bool(stop_on_terminal_status and status_value in {"success", "failure"}),
            failed_unsent_delta=0,
        )

    try:
        if webhook_url is None:
            raise NotifyConfigError("webhook URL is required when not running in --dry-run mode")
        post_with_backoff(
            webhook_url,
            formatted_payload,
            tls_ca_bundle=resolved_tls_ca_bundle,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retry_max=retry_max,
            retry_base_seconds=retry_base_seconds,
        )
        sent_or_spooled = True
    except NotifyDeliveryError:
        if spool_dir_value is not None:
            _spool_payload(spool_dir_value, provider=provider_value, payload=payload)
            sent_or_spooled = True
        elif fail_fast:
            raise
        else:
            sent_or_spooled = False

    if not sent_or_spooled:
        return DeliveryOutcome(cursor_advanced=False, terminal_reached=False, failed_unsent_delta=1)
    return DeliveryOutcome(
        cursor_advanced=True,
        terminal_reached=bool(stop_on_terminal_status and status_value in {"success", "failure"}),
        failed_unsent_delta=0,
    )
