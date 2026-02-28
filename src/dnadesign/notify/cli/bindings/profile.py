"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/bindings/profile.py

Profile command binding implementations for notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer


def run_profile_init_impl(
    *,
    deps: Any,
    profile: Path,
    provider: str,
    url_env: str,
    events: Path,
    cursor: Path | None,
    only_actions: str | None,
    only_tools: str | None,
    spool_dir: Path | None,
    include_args: bool,
    include_context: bool,
    include_raw_event: bool,
    progress_step_pct: int | None,
    progress_min_seconds: float | None,
    tls_ca_bundle: Path | None,
    policy: str | None,
    force: bool,
) -> None:
    deps.run_profile_init_command(
        profile=profile,
        provider=provider,
        url_env=url_env,
        events=events,
        cursor=cursor,
        only_actions=only_actions,
        only_tools=only_tools,
        spool_dir=spool_dir,
        include_args=include_args,
        include_context=include_context,
        include_raw_event=include_raw_event,
        progress_step_pct=progress_step_pct,
        progress_min_seconds=progress_min_seconds,
        tls_ca_bundle=tls_ca_bundle,
        policy=policy,
        force=force,
        resolve_usr_events_path_fn=deps._resolve_usr_events_path,
        resolve_workflow_policy_fn=deps._resolve_workflow_policy,
        policy_defaults_fn=deps._policy_defaults_for,
        resolve_existing_file_path_fn=deps._resolve_existing_file_path,
        write_profile_file_fn=lambda path, payload, overwrite: deps._write_profile_file(path, payload, force=overwrite),
    )


def run_profile_wizard_impl(
    *,
    deps: Any,
    ctx: typer.Context,
    profile: Path,
    provider: str,
    events: Path,
    cursor: Path | None,
    only_actions: str | None,
    only_tools: str | None,
    spool_dir: Path | None,
    include_args: bool,
    include_context: bool,
    include_raw_event: bool,
    progress_step_pct: int | None,
    progress_min_seconds: float | None,
    tls_ca_bundle: Path | None,
    policy: str | None,
    secret_source: str,
    url_env: str | None,
    secret_ref: str | None,
    webhook_url: str | None,
    store_webhook: bool,
    json_output: bool,
    force: bool,
) -> None:
    deps.run_profile_wizard_command(
        ctx=ctx,
        profile=profile,
        provider=provider,
        events=events,
        cursor=cursor,
        only_actions=only_actions,
        only_tools=only_tools,
        spool_dir=spool_dir,
        include_args=include_args,
        include_context=include_context,
        include_raw_event=include_raw_event,
        progress_step_pct=progress_step_pct,
        progress_min_seconds=progress_min_seconds,
        tls_ca_bundle=tls_ca_bundle,
        policy=policy,
        secret_source=secret_source,
        url_env=url_env,
        secret_ref=secret_ref,
        webhook_url=webhook_url,
        store_webhook=store_webhook,
        json_output=json_output,
        force=force,
        default_profile_path=deps._DEFAULT_PROFILE_PATH,
        resolve_profile_path_for_wizard_fn=deps._resolve_profile_path_for_wizard,
        create_wizard_profile_flow_fn=deps._create_wizard_profile_flow,
        resolve_usr_events_path_fn=deps._resolve_usr_events_path,
        ensure_private_directory_fn=deps._ensure_private_directory,
        secret_backend_available_fn=deps.is_secret_backend_available,
        resolve_secret_ref_fn=deps.resolve_secret_ref,
        store_secret_ref_fn=deps.store_secret_ref,
        write_profile_file_fn=lambda path, payload, overwrite: deps._write_profile_file(path, payload, force=overwrite),
    )


def run_profile_show_impl(
    *,
    deps: Any,
    profile: Path,
) -> None:
    deps.run_profile_show_command(
        profile=profile,
        read_profile_fn=deps._read_profile,
    )


def run_profile_doctor_impl(
    *,
    deps: Any,
    profile: Path,
    json_output: bool,
) -> None:
    deps.run_profile_doctor_command(
        profile=profile,
        json_output=json_output,
        read_profile_fn=deps._read_profile,
        resolve_profile_webhook_source_fn=deps._resolve_profile_webhook_source,
        resolve_path_value_fn=deps._resolve_path_value,
        resolve_profile_events_source_fn=deps._resolve_profile_events_source,
        resolve_usr_events_path_fn=deps._resolve_usr_events_path,
        resolve_optional_path_value_fn=deps._resolve_optional_path_value,
        probe_path_writable_fn=deps._probe_path_writable,
        resolve_webhook_url_fn=deps.resolve_webhook_url,
        validate_provider_webhook_url_fn=deps.validate_provider_webhook_url,
        resolve_tls_ca_bundle_fn=deps.resolve_tls_ca_bundle,
    )
