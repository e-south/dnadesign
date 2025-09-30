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
    key: str = typer.Argument(
        None, help="ID or sequence (positional). Use --id/--sequence to disambiguate."
    ),
    id: str = typer.Option(None, "--id"),
    sequence: str = typer.Option(None, "--sequence"),
    with_sequence: bool = typer.Option(True, "--with-sequence/--no-sequence"),
    json: bool = typer.Option(False, "--json"),
):
    try:
        cfg = load_cli_config(config)
        store = store_from_cfg(cfg)
        df = store.load()
        # Allow positional key if neither flag is present
        if not id and not sequence:
            if key:
                # Heuristic: prefer exact id match; else exact sequence match
                sid = df["id"].astype(str)
                if (sid == key).any():
                    id = key
                elif "sequence" in df.columns and (df["sequence"] == key).any():
                    sequence = key
                else:
                    raise OpalError(
                        "Record not found by positional key. Use --id or --sequence.",
                        ExitCodes.BAD_ARGS,
                    )
            else:
                raise OpalError(
                    "Provide --id or --sequence (or positional key).",
                    ExitCodes.BAD_ARGS,
                )
        # Default events path under campaign workdir
        base = Path(cfg.campaign.workdir) / "outputs"
        typed_pred = base / "events" / "run_pred.parquet"
        ev_path = typed_pred if typed_pred.exists() else (base / "events.parquet")
        report = build_record_report(
            df,
            cfg.campaign.slug,
            id_=id,
            sequence=sequence,
            with_sequence=with_sequence,
            events_path=ev_path if ev_path.exists() else None,
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
