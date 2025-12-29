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

from ...ledger import LedgerReader
from ...record_show import build_record_report
from ...utils import ExitCodes, OpalError, print_stdout
from ...workspace import CampaignWorkspace
from ..formatting import render_record_report_human
from ..registry import cli_command
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
    resolve_config_path,
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
    legacy: bool = typer.Option(False, "--legacy", help="Read from legacy events.parquet (deprecated)."),
    json: bool = typer.Option(False, "--json"),
):
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(config)
        store = store_from_cfg(cfg)
        df = store.load()
        if id and sequence:
            raise OpalError("Provide only one of --id or --sequence (not both).")
        if id is None and sequence is None:
            if not key:
                raise OpalError("Provide a record id or sequence.")
            # Try id match first, then sequence
            id_match = df["id"].astype(str) == str(key)
            seq_match = df["sequence"].astype(str) == str(key) if "sequence" in df.columns else None
            if id_match.any() and seq_match is not None and seq_match.any():
                raise OpalError("Key matches both an id and a sequence; use --id or --sequence.")
            if id_match.any():
                id = str(key)
            elif seq_match is not None and seq_match.any():
                sequence = str(key)
            else:
                raise OpalError("Record not found for key; use --id or --sequence explicitly.")

        ws = CampaignWorkspace.from_config(cfg, cfg_path)
        ledger_reader = None
        legacy_path = None
        if legacy:
            legacy_path = ws.outputs_dir / "events.parquet"
            if not legacy_path.exists():
                raise OpalError(f"Legacy events.parquet not found at {legacy_path}.")
        else:
            ledger_reader = LedgerReader(ws)

        report = build_record_report(
            df,
            cfg.campaign.slug,
            id_=id,
            sequence=sequence,
            with_sequence=with_sequence,
            ledger_reader=ledger_reader,
            legacy_events_path=legacy_path,
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
