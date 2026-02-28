"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/handlers/profile/init_cmd.py

Execution logic for notify profile init command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import typer

from ....errors import NotifyError
from ....profiles.schema import PROFILE_VERSION


def run_profile_init_command(
    *,
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
    resolve_usr_events_path_fn: Callable[..., Path],
    resolve_workflow_policy_fn: Callable[[str | None], str | None],
    policy_defaults_fn: Callable[[str], dict[str, Any]],
    resolve_existing_file_path_fn: Callable[..., Path],
    write_profile_file_fn: Callable[[Path, dict[str, Any], bool], None],
) -> None:
    try:
        profile_path = profile.expanduser().resolve()
        events_path = resolve_usr_events_path_fn(events)
        policy_name = resolve_workflow_policy_fn(policy=policy)

        cursor_path = cursor.expanduser().resolve() if cursor is not None else (events_path.parent / "notify.cursor")
        payload: dict[str, Any] = {
            "profile_version": PROFILE_VERSION,
            "provider": str(provider).strip(),
            "events": str(events_path),
            "cursor": str(cursor_path),
            "include_args": bool(include_args),
            "include_context": bool(include_context),
            "include_raw_event": bool(include_raw_event),
            "webhook": {"source": "env", "ref": str(url_env).strip()},
        }
        if progress_step_pct is not None:
            progress_step_pct_value = int(progress_step_pct)
            if progress_step_pct_value < 1 or progress_step_pct_value > 100:
                raise NotifyError("progress_step_pct must be an integer between 1 and 100")
            payload["progress_step_pct"] = progress_step_pct_value
        if progress_min_seconds is not None:
            progress_min_seconds_value = float(progress_min_seconds)
            if progress_min_seconds_value <= 0.0:
                raise NotifyError("progress_min_seconds must be a positive number")
            payload["progress_min_seconds"] = progress_min_seconds_value
        if only_actions is not None:
            payload["only_actions"] = str(only_actions).strip()
        if only_tools is not None:
            payload["only_tools"] = str(only_tools).strip()
        if spool_dir is not None:
            payload["spool_dir"] = str(spool_dir.expanduser().resolve())
        if tls_ca_bundle is not None:
            payload["tls_ca_bundle"] = str(
                resolve_existing_file_path_fn(field="tls_ca_bundle", path_value=tls_ca_bundle)
            )
        if policy_name is not None:
            payload["policy"] = policy_name
            for key, value in policy_defaults_fn(policy_name).items():
                payload.setdefault(key, value)

        if not payload["provider"]:
            raise NotifyError("provider must be a non-empty string")
        if not payload["webhook"]["ref"]:
            raise NotifyError("url_env must be a non-empty string")

        write_profile_file_fn(profile_path, payload, force)
        typer.echo(f"Profile written: {profile_path}")
    except NotifyError as exc:
        typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)
