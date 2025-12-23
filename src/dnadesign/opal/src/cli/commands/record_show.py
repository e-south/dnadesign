"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/record_show.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from ...record_show import build_record_report
from ...utils import ExitCodes, OpalError, print_stdout
from ..formatting import render_record_report_human
from ..registry import cli_command
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
    store_from_cfg,
)


@cli_command(
    "record-show",
    help="Per-record report: ground truth & history; per-round predictions/ranks.",
)
def cmd_record_show(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    key: str = typer.Argument(None, help="ID or sequence (positional). Use --id/--sequence to disambiguate."),
    id: str = typer.Option(None, "--id"),
    sequence: str = typer.Option(None, "--sequence"),
    with_sequence: bool = typer.Option(True, "--with-sequence/--no-sequence"),
    json: bool = typer.Option(False, "--json"),
):
    try:
        cfg = load_cli_config(config)
        store = store_from_cfg(cfg)
        df = store.load()
        # ... (id/sequence resolution unchanged) ...

        base = Path(cfg.campaign.workdir) / "outputs"

        # Prefer new typed sinks (consolidated file, then directory), else legacy
        candidates = [
            base / "ledger.predictions.parquet",
            base / "ledger.predictions",
            base / "events" / "run_pred.parquet",  # legacy
            base / "events.parquet",  # legacy
        ]
        ev_path = next((p for p in candidates if p.exists()), None)

        report = build_record_report(
            df,
            cfg.campaign.slug,
            id_=id,
            sequence=sequence,
            with_sequence=with_sequence,
            events_path=ev_path if ev_path else None,
            records_path=store.records_path,
        )
        if json:
            json_out(report)
        else:
            print_stdout(render_record_report_human(report))
    except OpalError as e:
        opal_error("run", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("record-show", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
