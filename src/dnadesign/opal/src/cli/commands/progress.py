"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/progress.py

CLI command for machine-readable campaign progress summaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...core.utils import ExitCodes, OpalError, print_stdout
from ..registry import cli_command
from ._common import internal_error, json_error, json_out, opal_error


@cli_command("progress", help="Summarize campaign round progress from state and round logs.")
def cmd_progress(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="campaign.yaml or campaign directory",
        envvar="OPAL_CONFIG",
    ),
    round: Optional[str] = typer.Option("latest", "--round", "-r", help="Round selector: int, latest, or all."),
    json: bool = typer.Option(False, "--json/--text", help="Output format."),
) -> None:
    try:
        from ...reporting.progress import build_campaign_progress, render_campaign_progress_text

        payload = build_campaign_progress(config, round_selector=round)
        if json:
            json_out(payload)
        else:
            print_stdout(render_campaign_progress_text(payload))
    except OpalError as e:
        if json:
            json_error("progress", e)
        else:
            opal_error("progress", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("progress", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
