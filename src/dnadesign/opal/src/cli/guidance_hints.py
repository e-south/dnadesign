"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/guidance_hints.py

Shared helpers for human-readable next-step hints printed by CLI commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.utils import print_stdout


def _print_hint_block(lines: list[str]) -> None:
    if not lines:
        return
    print_stdout("")
    print_stdout("Next steps")
    for line in lines:
        print_stdout(f"- {line}")


def maybe_print_hints(
    *,
    command_name: str,
    cfg_path: Path,
    no_hints: bool,
    json_output: bool,
    observed_round: int | None = None,
    labels_as_of: int | None = None,
    explain_info: dict[str, Any] | None = None,
) -> None:
    if no_hints or json_output:
        return
    cfg = str(Path(cfg_path).resolve())
    lines: list[str] = []
    if command_name == "init":
        lines = [
            f"opal validate -c {cfg}",
            f"opal ingest-y -c {cfg} --observed-round 0 --in <labels.xlsx> --apply",
            f"opal guide next -c {cfg} --labels-as-of 0",
        ]
    elif command_name == "validate":
        lines = [
            f"opal ingest-y -c {cfg} --observed-round 0 --in <labels.xlsx> --apply",
            f"opal guide -c {cfg} --format markdown",
        ]
    elif command_name == "ingest":
        rr = int(observed_round) if observed_round is not None else 0
        lines = [
            f"opal run -c {cfg} --labels-as-of {rr}",
            f"opal explain -c {cfg} --labels-as-of {rr + 1}",
        ]
    elif command_name == "run":
        rr = int(labels_as_of) if labels_as_of is not None else "latest"
        lines = [
            f"opal verify-outputs -c {cfg} --round latest",
            f"opal ctx audit -c {cfg} --round latest",
            f"opal status -c {cfg} --round {rr}",
        ]
    elif command_name == "verify-outputs":
        lines = [
            f"opal runs list -c {cfg}",
            f"opal record-show -c {cfg} --id <selected_id> --run-id latest",
        ]
    elif command_name == "explain":
        rr = int(labels_as_of) if labels_as_of is not None else 0
        lines = [
            f"opal ingest-y -c {cfg} --observed-round {rr} --in <labels.xlsx> --apply",
            f"opal run -c {cfg} --labels-as-of {rr}",
        ]
        if explain_info:
            preflight = dict(explain_info.get("preflight") or {})
            if bool(preflight.get("sfxi_run_will_fail")):
                lines.insert(
                    0,
                    f"SFXI preflight indicates round {rr} is under min_n; ingest labels for observed round {rr} first.",
                )
    _print_hint_block(lines)
