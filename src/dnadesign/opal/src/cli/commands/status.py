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
from typing import Optional

import typer

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
        ledger_reader = None
        if with_ledger:
            from ...storage.ledger import LedgerReader
            from ...storage.workspace import CampaignWorkspace

            ws = CampaignWorkspace.from_config(cfg, cfg_path)
            ledger_reader = LedgerReader(ws)
        round_sel = None if round is None else str(round).strip().lower()
        if round_sel in (None, "", "latest", "unspecified"):
            round_k = None
        else:
            try:
                round_k = int(round_sel)
            except Exception as e:
                raise OpalError("Invalid --round: must be an integer or 'latest'.") from e
        st = build_status(
            Path(cfg.campaign.workdir) / "state.json",
            round_k=round_k,
            show_all=all,
            ledger_reader=ledger_reader,
            include_ledger=with_ledger,
        )
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
