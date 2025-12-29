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
    print_config_context,
    resolve_config_path,
    store_from_cfg,
)


@cli_command("explain", help="Dry-run planner for a round; prints counts and plan (no writes).")
def cmd_explain(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG", help="campaign.yaml"),
    round: int = typer.Option(
        ...,
        "--round",
        "-r",
        "--labels-as-of",
        help="Labels cutoff (same as run --labels-as-of).",
    ),
    json: bool = typer.Option(
        False,
        "--json/--human",
        help="Output as JSON (default: human).",
    ),
):
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        store = store_from_cfg(cfg)
        df = store.load()
        info = explain_round(store, df, cfg, round)
        if json:
            json_out(info)
        else:
            print_config_context(cfg_path, cfg=cfg, records_path=store.records_path)
            print_stdout(render_explain_human(info))
    except OpalError as e:
        opal_error("explain", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("explain", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
