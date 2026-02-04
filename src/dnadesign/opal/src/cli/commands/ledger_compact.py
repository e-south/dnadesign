"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/ledger_compact.py

Compacts ledger datasets to remove duplicate entries. Provides a CLI helper
for run_meta cleanup.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...core.utils import ExitCodes, OpalError, print_stdout
from ...storage.ledger import compact_runs_ledger
from ...storage.locks import CampaignLock
from ...storage.workspace import CampaignWorkspace
from ..formatting import kv_block
from ..registry import cli_command
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
    print_config_context,
    prompt_confirm,
    resolve_config_path,
)


@cli_command(
    "ledger-compact",
    help="Compact ledger datasets (run_meta).",
)
def cmd_ledger_compact(
    config: Optional[Path] = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    runs: bool = typer.Option(False, "--runs", help="Compact run_meta ledger (runs.parquet)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive confirmation."),
    json: bool = typer.Option(False, "--json/--human", help="Output format."),
) -> None:
    try:
        if not runs:
            raise OpalError("Nothing to compact: pass --runs.", ExitCodes.BAD_ARGS)

        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        ws = CampaignWorkspace.from_config(cfg, cfg_path)

        if not json:
            print_config_context(cfg_path, cfg=cfg)
            print_stdout(kv_block("ledger-compact target", {"runs": str(ws.ledger_runs_path)}))

        if not yes:
            if not prompt_confirm(
                "Proceed with ledger-compact? This rewrites ledger datasets. (y/N): ",
                non_interactive_hint="No TTY available. Re-run with --yes to confirm.",
            ):
                print_stdout("Aborted.")
                raise typer.Exit(code=ExitCodes.BAD_ARGS)

        results = {}
        with CampaignLock(ws.workdir):
            if runs:
                results["runs"] = compact_runs_ledger(ws.ledger_runs_path)

        out = {"ok": True, **results}
        if json:
            json_out(out)
        else:
            print_stdout(kv_block("ledger-compact", out))
    except OpalError as e:
        opal_error("ledger-compact", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("ledger-compact", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
