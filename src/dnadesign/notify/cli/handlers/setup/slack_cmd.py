"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/handlers/setup/slack_cmd.py

Execution logic for notify setup slack command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import typer

from ....errors import NotifyError


def run_setup_slack_command(
    *,
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
    resolve_setup_events_fn: Callable[..., Any],
    resolve_tool_events_path_fn: Callable[..., tuple[Path, str | None]],
    resolve_tool_workspace_config_fn: Callable[..., Path],
    normalize_tool_name_fn: Callable[[str | None], str | None],
    resolve_profile_path_for_setup_fn: Callable[..., Path],
    create_wizard_profile_flow_fn: Callable[..., dict[str, Any]],
    resolve_usr_events_path_fn: Callable[..., Path],
    ensure_private_directory_fn: Callable[..., None],
    secret_backend_available_fn: Callable[[str], bool],
    resolve_secret_ref_fn: Callable[[str], str],
    store_secret_ref_fn: Callable[[str, str], str],
    write_profile_file_fn: Callable[[Path, dict[str, Any], bool], None],
) -> None:
    try:
        setup_resolution = resolve_setup_events_fn(
            events=events,
            tool=tool,
            config=config,
            workspace=workspace,
            policy=policy,
            search_start=Path.cwd(),
            resolve_tool_events_path_fn=resolve_tool_events_path_fn,
            resolve_tool_workspace_config_fn=resolve_tool_workspace_config_fn,
            normalize_tool_name_fn=normalize_tool_name_fn,
        )
        resolved_config: Path | None = None
        if setup_resolution.events_source is not None:
            config_value = setup_resolution.events_source.get("config")
            if config_value is not None:
                resolved_config = Path(str(config_value))

        profile_value = resolve_profile_path_for_setup_fn(
            profile=profile,
            tool_name=setup_resolution.tool_name,
            policy=setup_resolution.policy,
            config=resolved_config,
        )

        result = create_wizard_profile_flow_fn(
            profile=profile_value,
            provider="slack",
            events=setup_resolution.events_path,
            cursor=cursor,
            only_actions=None,
            only_tools=None,
            spool_dir=spool_dir,
            include_args=include_args,
            include_context=include_context,
            include_raw_event=include_raw_event,
            progress_step_pct=progress_step_pct,
            progress_min_seconds=progress_min_seconds,
            tls_ca_bundle=tls_ca_bundle,
            policy=setup_resolution.policy,
            secret_source=secret_source,
            url_env=url_env,
            secret_ref=secret_ref,
            webhook_url=webhook_url,
            store_webhook=store_webhook,
            force=force,
            events_require_exists=setup_resolution.events_require_exists,
            events_source=setup_resolution.events_source,
            resolve_events_path=lambda path, required: resolve_usr_events_path_fn(path, require_exists=required),
            ensure_private_directory_fn=ensure_private_directory_fn,
            secret_backend_available_fn=secret_backend_available_fn,
            resolve_secret_ref_fn=resolve_secret_ref_fn,
            store_secret_ref_fn=store_secret_ref_fn,
            write_profile_file_fn=lambda path, payload, overwrite: write_profile_file_fn(path, payload, overwrite),
        )
        if json_output:
            typer.echo(json.dumps({"ok": True, **result}, sort_keys=True))
            return
        typer.echo(f"Profile written: {result['profile']}")
        for line in result["next_steps"]:
            typer.echo(line)
    except NotifyError as exc:
        if json_output:
            typer.echo(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)
