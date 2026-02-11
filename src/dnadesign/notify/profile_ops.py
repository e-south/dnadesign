"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/profile_ops.py

Profile setup helper functions shared by notify CLI commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def sanitize_profile_name(profile_path: Path) -> str:
    stem = str(profile_path.stem or "").strip()
    cleaned = "".join(char if char.isalnum() else "-" for char in stem).strip("-")
    return cleaned or "default"


def wizard_next_steps(*, profile_path: Path, webhook_config: dict[str, str], events_exists: bool) -> list[str]:
    profile_arg = str(profile_path)
    steps = [
        "Next steps:",
    ]
    if events_exists:
        steps.extend(
            [
                f"  1) uv run notify profile doctor --profile {profile_arg}",
                f"  2) uv run notify usr-events watch --profile {profile_arg} --dry-run",
                f"  3) uv run notify usr-events watch --profile {profile_arg} --follow",
            ]
        )
    else:
        steps.extend(
            [
                "  1) Submit your tool run (events path is reserved but not created yet).",
                (
                    f"  2) uv run notify usr-events watch --profile {profile_arg} --follow "
                    "--wait-for-events --stop-on-terminal-status --idle-timeout 900"
                ),
            ]
        )
    steps.extend(
        [
            "  batch: qsub -P <project> \\",
            f"    -v NOTIFY_PROFILE={profile_arg} \\",
            "    docs/hpc/jobs/bu_scc_notify_watch.qsub",
        ]
    )
    source = str(webhook_config.get("source") or "").strip().lower()
    ref = str(webhook_config.get("ref") or "").strip()
    if source == "env" and ref:
        steps.append(f"  env: export {ref}=<your_webhook_url>")
    return steps
