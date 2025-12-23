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

from ...explain import explain_round
from ...utils import ExitCodes, OpalError, print_stdout
from ..formatting import render_explain_human
from ..registry import cli_command
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
    store_from_cfg,
)


@cli_command("explain", help="Dry-run planner for a round; prints counts and plan (no writes).")
def cmd_explain(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG", help="campaign.yaml"),
    round: int = typer.Option(..., "--round", "-r"),
    json: bool = typer.Option(
        False,
        "--json/--human",
        help="Output as JSON (default: human).",
    ),
    format: str = typer.Option(  # deprecated alias
        None,
        "--format",
        "-f",
        help="(deprecated) Use --json/--human instead. Allowed: 'json' or 'human'.",
        case_sensitive=False,
    ),
):
    try:
        cfg = load_cli_config(config)
        store = store_from_cfg(cfg)
        df = store.load()
        info = explain_round(store, df, cfg, round)
        fmt = str(format).lower() if format else None
        if json or fmt == "json":
            json_out(info)
        else:
            print_stdout(render_explain_human(info))
    except OpalError as e:
        opal_error("run", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("explain", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
