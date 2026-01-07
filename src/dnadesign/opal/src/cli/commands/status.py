"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/status.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...core.rounds import resolve_round_index_from_state
from ...core.utils import ExitCodes, OpalError, print_stdout
from ...reporting.status import build_status
from ..formatting import render_status_human
from ..registry import cli_command
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
    print_config_context,
    resolve_config_path,
)


@cli_command("status", help="Dashboard from state.json (latest round by default).")
def cmd_status(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: Optional[str] = typer.Option(None, "--round", help="Round selector: int or 'latest'."),
    all: bool = typer.Option(False, "--all"),
    with_ledger: bool = typer.Option(False, "--with-ledger", help="Include ledger summaries in output."),
    json: bool = typer.Option(False, "--json"),
):
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        if not json:
            print_config_context(cfg_path, cfg=cfg)
        if all and round is not None:
            raise OpalError("Provide only one of --all or --round.")
        ledger_reader = None
        from ...storage.workspace import CampaignWorkspace

        ws = CampaignWorkspace.from_config(cfg, cfg_path)
        if with_ledger:
            from ...storage.ledger import LedgerReader

            ledger_reader = LedgerReader(ws)
        round_k = resolve_round_index_from_state(ws.state_path, round) if round is not None else None
        st = build_status(
            ws.state_path,
            round_k=round_k,
            show_all=all,
            ledger_reader=ledger_reader,
            include_ledger=with_ledger,
        )
        if "error" in st:
            raise OpalError(str(st["error"]))
        if json or all:
            json_out(st)
        else:
            print_stdout(render_status_human(st))
    except OpalError as e:
        opal_error("status", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("status", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
