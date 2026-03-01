"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/bindings/__init__.py

Dependency-wired command bindings for notify CLI groups.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import random
import sys
from functools import partial
from pathlib import Path

import typer

from ...errors import NotifyDeliveryError
from . import deps as _deps
from . import helpers as _helpers
from .deps import DEPENDENCY_EXPORTS
from .profile import (
    run_profile_doctor_impl,
    run_profile_init_impl,
    run_profile_show_impl,
    run_profile_wizard_impl,
)
from .registry import register_notify_cli_bindings as _register_notify_cli_bindings
from .runtime import run_spool_drain_impl, run_usr_events_watch_impl
from .send import run_send_impl
from .setup import (
    run_setup_list_workspaces_impl,
    run_setup_resolve_events_impl,
    run_setup_slack_impl,
    run_setup_webhook_impl,
)

for _export in DEPENDENCY_EXPORTS:
    globals()[_export] = getattr(_deps, _export)
del _export


def _post_with_backoff(
    webhook_url: str,
    formatted_payload: dict[str, object],
    *,
    tls_ca_bundle: Path | None,
    connect_timeout: float,
    read_timeout: float,
    retry_max: int,
    retry_base_seconds: float,
) -> None:
    module = sys.modules[__name__]
    post_json_fn = getattr(module, "post_json")
    time_module = getattr(module, "time")
    _helpers.post_with_backoff(
        webhook_url,
        formatted_payload,
        tls_ca_bundle=tls_ca_bundle,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        retry_max=retry_max,
        retry_base_seconds=retry_base_seconds,
        post_json_fn=post_json_fn,
        notify_delivery_error_cls=NotifyDeliveryError,
        sleep_fn=time_module.sleep,
        jitter_fn=random.uniform,
    )


_DEPS = sys.modules[__name__]

_send_impl = partial(run_send_impl, deps=_DEPS)
_profile_init_impl = partial(run_profile_init_impl, deps=_DEPS)
_profile_wizard_impl = partial(run_profile_wizard_impl, deps=_DEPS)
_profile_show_impl = partial(run_profile_show_impl, deps=_DEPS)
_profile_doctor_impl = partial(run_profile_doctor_impl, deps=_DEPS)
_setup_slack_impl = partial(run_setup_slack_impl, deps=_DEPS)
_setup_webhook_impl = partial(run_setup_webhook_impl, deps=_DEPS)
_setup_resolve_events_impl = partial(run_setup_resolve_events_impl, deps=_DEPS)
_setup_list_workspaces_impl = partial(run_setup_list_workspaces_impl, deps=_DEPS)
_usr_events_watch_impl = partial(run_usr_events_watch_impl, deps=_DEPS)
_spool_drain_impl = partial(run_spool_drain_impl, deps=_DEPS)


def register_notify_cli_bindings(
    *,
    app: typer.Typer,
    usr_events_app: typer.Typer,
    spool_app: typer.Typer,
    profile_app: typer.Typer,
    setup_app: typer.Typer,
) -> None:
    _register_notify_cli_bindings(
        app=app,
        usr_events_app=usr_events_app,
        spool_app=spool_app,
        profile_app=profile_app,
        setup_app=setup_app,
        usr_events_watch_handler=_usr_events_watch_impl,
        spool_drain_handler=_spool_drain_impl,
        send_handler=_send_impl,
        profile_init_handler=_profile_init_impl,
        profile_wizard_handler=_profile_wizard_impl,
        profile_show_handler=_profile_show_impl,
        profile_doctor_handler=_profile_doctor_impl,
        setup_slack_handler=_setup_slack_impl,
        setup_webhook_handler=_setup_webhook_impl,
        setup_resolve_events_handler=_setup_resolve_events_impl,
        setup_list_workspaces_handler=_setup_list_workspaces_impl,
    )


__all__ = ["register_notify_cli_bindings", *DEPENDENCY_EXPORTS]
