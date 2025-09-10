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

from ...config import load_config
from ...status import build_status
from ...utils import ExitCodes, OpalError, print_stdout
from ..registry import cli_command
from ._common import internal_error, json_out, resolve_config_path


@cli_command("status", help="Dashboard from state.json (latest round by default).")
def cmd_status(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: int = typer.Option(None, "--round"),
    all: bool = typer.Option(False, "--all"),
    json: bool = typer.Option(False, "--json"),
):
    try:
        cfg = load_config(resolve_config_path(config))
        st = build_status(
            Path(cfg.campaign.workdir) / "state.json", round_k=round, show_all=all
        )
        if json or all:
            json_out(st)
        else:
            if st.get("latest_round"):
                lr = st["latest_round"]
                print_stdout(
                    f"latest round: r={lr['round_index']}, n_train={lr['number_of_training_examples_used_in_round']}, \
                        n_scored={lr['number_of_candidates_scored_in_round']}"
                )
            else:
                print_stdout("No completed rounds.")
    except OpalError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("status", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
