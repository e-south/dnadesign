"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/explain.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from ...config import load_config
from ...explain import explain_round
from ...utils import ExitCodes, OpalError
from ..registry import cli_command
from ._common import internal_error, json_out, resolve_config_path, store_from_cfg


@cli_command(
    "explain", help="Dry-run planner for a round; prints counts and plan (no writes)."
)
def cmd_explain(
    config: Path = typer.Option(
        None, "--config", "-c", envvar="OPAL_CONFIG", help="campaign.yaml"
    ),
    round: int = typer.Option(..., "--round", "-r"),
):
    try:
        cfg = load_config(resolve_config_path(config))
        store = store_from_cfg(cfg)
        df = store.load()
        info = explain_round(store, df, cfg, round)
        json_out(info)
    except OpalError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("explain", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
