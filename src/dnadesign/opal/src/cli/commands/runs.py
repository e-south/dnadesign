"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/runs.py

Inspect ledger run_meta entries (list/show).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...core.rounds import resolve_round_index_from_runs
from ...core.utils import ExitCodes, OpalError, print_stdout
from ...reporting.summary import list_runs, select_run_meta, summarize_run_meta
from ...storage.ledger import LedgerReader
from ...storage.workspace import CampaignWorkspace
from ..formatting import render_run_meta_human, render_runs_list_human
from ..registry import cli_group
from ._common import internal_error, json_out, load_cli_config, opal_error, print_config_context, resolve_config_path

runs_app = typer.Typer(no_args_is_help=True, help="Inspect ledger run_meta entries.")
cli_group("runs", help="Inspect ledger run_meta entries.")(runs_app)


@runs_app.command("list", help="List run_meta entries (optionally filtered by round).")
def runs_list(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: Optional[str] = typer.Option(None, "--round", "-r", help="Round selector: int or 'latest'."),
    json: bool = typer.Option(False, "--json/--human", help="Output format (default: human)."),
) -> None:
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        ws = CampaignWorkspace.from_config(cfg, cfg_path)
        reader = LedgerReader(ws)
        runs_df = list_runs(reader)
        round_sel = resolve_round_index_from_runs(runs_df, round, allow_none=True)
        if round_sel is not None:
            runs_df = runs_df[runs_df["as_of_round"] == int(round_sel)]
        if json:
            json_out(runs_df.to_dict(orient="records"))
        else:
            print_config_context(cfg_path, cfg=cfg)
            summary_rows = [summarize_run_meta(r) for _, r in runs_df.iterrows()]
            print_stdout(render_runs_list_human(summary_rows))
    except OpalError as e:
        opal_error("runs list", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("runs list", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@runs_app.command("show", help="Show a specific run_meta entry (by run_id or round).")
def runs_show(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: Optional[str] = typer.Option(None, "--round", "-r", help="Round selector: int or 'latest'."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Explicit run_id to display."),
    json: bool = typer.Option(False, "--json/--human", help="Output format (default: human)."),
) -> None:
    try:
        if round and run_id:
            raise OpalError("Provide only one of --round or --run-id.")
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        ws = CampaignWorkspace.from_config(cfg, cfg_path)
        reader = LedgerReader(ws)
        runs_df = reader.read_runs()
        round_sel = resolve_round_index_from_runs(runs_df, round) if run_id is None else None
        row = select_run_meta(runs_df, round_sel=round_sel, run_id=run_id)
        if json:
            json_out(row.to_dict())
        else:
            print_config_context(cfg_path, cfg=cfg)
            print_stdout(render_run_meta_human(row.to_dict()))
    except OpalError as e:
        opal_error("runs show", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("runs show", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
