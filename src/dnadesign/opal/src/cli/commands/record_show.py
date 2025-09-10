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

from ...config import load_config
from ...record_show import build_record_report
from ...utils import ExitCodes, OpalError, print_stdout
from ..registry import cli_command
from ._common import internal_error, json_out, resolve_config_path, store_from_cfg


@cli_command(
    "record-show",
    help="Per-record report: ground truth & history; per-round predictions/ranks.",
)
def cmd_record_show(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    id: str = typer.Option(None, "--id"),
    sequence: str = typer.Option(None, "--sequence"),
    with_sequence: bool = typer.Option(False, "--with-sequence"),
    json: bool = typer.Option(False, "--json"),
):
    try:
        cfg = load_config(resolve_config_path(config))
        store = store_from_cfg(cfg)
        df = store.load()
        if not id and not sequence:
            raise OpalError("Provide --id or --sequence.", ExitCodes.BAD_ARGS)
        report = build_record_report(
            df,
            cfg.campaign.slug,
            id_=id,
            sequence=sequence,
            with_sequence=with_sequence,
        )
        if json:
            json_out(report)
        else:
            print_stdout("\n".join([f"{k}: {v}" for k, v in report.items()]))
    except OpalError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("record-show", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
