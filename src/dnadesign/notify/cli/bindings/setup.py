"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/bindings/setup.py

Setup command binding implementations for notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_setup_slack_impl(
    *,
    deps: Any,
    profile: Path,
    events: Path | None,
    tool: str | None,
    config: Path | None,
    workspace: str | None,
    policy: str | None,
    cursor: Path | None,
    spool_dir: Path | None,
    include_args: bool,
    include_context: bool,
    include_raw_event: bool,
    progress_step_pct: int | None,
    progress_min_seconds: float | None,
    tls_ca_bundle: Path | None,
    secret_source: str,
    url_env: str | None,
    secret_ref: str | None,
    webhook_url: str | None,
    store_webhook: bool,
    json_output: bool,
    force: bool,
) -> None:
    deps.run_setup_slack_command(
        profile=profile,
        events=events,
        tool=tool,
        config=config,
        workspace=workspace,
        policy=policy,
        cursor=cursor,
        spool_dir=spool_dir,
        include_args=include_args,
        include_context=include_context,
        include_raw_event=include_raw_event,
        progress_step_pct=progress_step_pct,
        progress_min_seconds=progress_min_seconds,
        tls_ca_bundle=tls_ca_bundle,
        secret_source=secret_source,
        url_env=url_env,
        secret_ref=secret_ref,
        webhook_url=webhook_url,
        store_webhook=store_webhook,
        json_output=json_output,
        force=force,
        resolve_setup_events_fn=deps._resolve_setup_events,
        resolve_tool_events_path_fn=deps._resolve_tool_events_path,
        resolve_tool_workspace_config_fn=deps._resolve_tool_workspace_config_path,
        normalize_tool_name_fn=deps._normalize_setup_tool_name,
        resolve_profile_path_for_setup_fn=deps._resolve_profile_path_for_setup,
        create_wizard_profile_flow_fn=deps._create_wizard_profile_flow,
        resolve_usr_events_path_fn=deps._resolve_usr_events_path,
        ensure_private_directory_fn=deps._ensure_private_directory,
        secret_backend_available_fn=deps.is_secret_backend_available,
        resolve_secret_ref_fn=deps.resolve_secret_ref,
        store_secret_ref_fn=deps.store_secret_ref,
        write_profile_file_fn=lambda path, payload, overwrite: deps._write_profile_file(path, payload, force=overwrite),
    )


def run_setup_webhook_impl(
    *,
    deps: Any,
    name: str,
    secret_source: str,
    url_env: str | None,
    secret_ref: str | None,
    webhook_url: str | None,
    store_webhook: bool,
    json_output: bool,
) -> None:
    deps.run_setup_webhook_command(
        name=name,
        secret_source=secret_source,
        url_env=url_env,
        secret_ref=secret_ref,
        webhook_url=webhook_url,
        store_webhook=store_webhook,
        json_output=json_output,
        resolve_cli_optional_string_fn=deps._resolve_cli_optional_string,
        resolve_webhook_config_fn=deps._resolve_webhook_config,
        secret_backend_available_fn=deps.is_secret_backend_available,
        resolve_secret_ref_fn=deps.resolve_secret_ref,
        store_secret_ref_fn=deps.store_secret_ref,
    )


def run_setup_resolve_events_impl(
    *,
    deps: Any,
    tool: str,
    config: Path | None,
    workspace: str | None,
    print_policy: bool,
    json_output: bool,
) -> None:
    deps.run_setup_resolve_events_command(
        tool=tool,
        config=config,
        workspace=workspace,
        print_policy=print_policy,
        json_output=json_output,
        normalize_tool_name_fn=deps._normalize_setup_tool_name,
        resolve_tool_workspace_config_fn=deps._resolve_tool_workspace_config_path,
        resolve_tool_events_path_fn=deps._resolve_tool_events_path,
    )


def run_setup_list_workspaces_impl(
    *,
    deps: Any,
    tool: str,
    json_output: bool,
) -> None:
    deps.run_setup_list_workspaces_command(
        tool=tool,
        json_output=json_output,
        list_tool_workspaces_fn=deps._list_tool_workspaces,
    )
