"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/status.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from ...status import build_status
from ...utils import ExitCodes, OpalError, print_stdout
from ..formatting import render_status_human
from ..registry import cli_command
from ._common import internal_error, json_out, load_cli_config, opal_error


@cli_command("status", help="Dashboard from state.json (latest round by default).")
def cmd_status(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: int = typer.Option(None, "--round"),
    all: bool = typer.Option(False, "--all"),
    json: bool = typer.Option(False, "--json"),
):
    try:
        cfg = load_cli_config(config)
        st = build_status(Path(cfg.campaign.workdir) / "state.json", round_k=round, show_all=all)
        if json or all:
            json_out(st)
        else:
            print_stdout(render_status_human(st))
    except OpalError as e:
        opal_error("run", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("status", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
