"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/profile_ops.py

Profile setup helper functions shared by notify CLI commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shlex
from pathlib import Path


def sanitize_profile_name(profile_path: Path) -> str:
    stem = str(profile_path.stem or "").strip()
    cleaned = "".join(char if char.isalnum() else "-" for char in stem).strip("-")
    return cleaned or "default"


def wizard_next_steps(
    *,
    profile_path: Path,
    webhook_config: dict[str, str],
    events_exists: bool,
    events_source: dict[str, str] | None = None,
) -> list[str]:
    profile_arg = str(profile_path)
    profile_arg_shell = shlex.quote(profile_arg)
    source_metadata = dict(events_source or {})
    source_tool = str(source_metadata.get("tool") or "").strip()
    source_config = str(source_metadata.get("config") or "").strip()
    source_config_shell = shlex.quote(source_config) if source_config else ""
    watch_with_tool_config = bool(source_tool and source_config)
    dry_run_watch = (
        f"uv run notify usr-events watch --tool {source_tool} --config {source_config_shell} --dry-run"
        if watch_with_tool_config
        else f"uv run notify usr-events watch --profile {profile_arg_shell} --dry-run"
    )
    follow_watch = (
        f"uv run notify usr-events watch --tool {source_tool} --config {source_config_shell} --follow"
        if watch_with_tool_config
        else f"uv run notify usr-events watch --profile {profile_arg_shell} --follow"
    )
    follow_wait_watch = (
        f"uv run notify usr-events watch --tool {source_tool} --config {source_config_shell} --follow "
        "--wait-for-events --stop-on-terminal-status --idle-timeout 900"
        if watch_with_tool_config
        else (
            f"uv run notify usr-events watch --profile {profile_arg_shell} --follow "
            "--wait-for-events --stop-on-terminal-status --idle-timeout 900"
        )
    )
    steps = [
        "Next steps:",
    ]
    if events_exists:
        steps.extend(
            [
                f"  1) uv run notify profile doctor --profile {profile_arg_shell}",
                f"  2) {dry_run_watch}",
                f"  3) {follow_watch}",
            ]
        )
    else:
        steps.extend(
            [
                "  1) Submit your tool run (events path is reserved but not created yet).",
                f"  2) {follow_wait_watch}",
            ]
        )
    steps.extend(
        [
            "  batch: qsub -P <project> \\",
            f"    -v NOTIFY_PROFILE={profile_arg_shell} \\",
            "    docs/bu-scc/jobs/notify-watch.qsub",
        ]
    )
    webhook_source = str(webhook_config.get("source") or "").strip().lower()
    ref = str(webhook_config.get("ref") or "").strip()
    if webhook_source == "env" and ref:
        steps.append(f"  env: export {ref}=<your_webhook_url>")
    return steps
