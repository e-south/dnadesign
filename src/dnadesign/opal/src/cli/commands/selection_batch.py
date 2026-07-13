"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/selection_batch.py

CLI for the deduplicated logical union of OPAL selection views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from ...core.utils import ExitCodes, OpalError, print_stdout, write_json
from ...reporting.selection_set import load_selection_batch
from ..registry import cli_group
from ._common import internal_error, json_error, json_out, opal_error, resolve_config_path

selection_batch_app = typer.Typer(no_args_is_help=True, help="Inspect and export a logical selection batch.")
cli_group("selection-batch", help="Inspect and export a logical selection batch.")(selection_batch_app)


@selection_batch_app.command("show", help="Show the deduplicated selection union for one OPAL run.")
def selection_batch_show(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: Optional[str] = typer.Option(None, "--round", "-r", help="Round selector: int or 'latest'."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Explicit run_id to show."),
    json: bool = typer.Option(False, "--json/--text", help="Output format (default: text)."),
) -> None:
    try:
        payload = load_selection_batch(resolve_config_path(config), round_selector=round, run_id=run_id)
        if json:
            json_out(payload)
            return
        print_stdout("selection-batch")
        print_stdout(
            f"- campaign: {payload['campaign']['slug']}  round: {payload['as_of_round']}  "
            f"run_id: {payload['run_id']}  unique: {payload['unique_count']}"
        )
        print_stdout(pd.DataFrame(payload["rows"]).to_string(index=False))
    except OpalError as exc:
        if json:
            json_error("selection-batch show", exc)
        else:
            opal_error("selection-batch show", exc)
        raise typer.Exit(code=exc.exit_code)
    except Exception as exc:
        internal_error("selection-batch show", exc)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@selection_batch_app.command("export", help="Export the deduplicated selection union for one OPAL run.")
def selection_batch_export(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: Optional[str] = typer.Option(None, "--round", "-r", help="Round selector: int or 'latest'."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Explicit run_id to export."),
    out: Path = typer.Option(..., "--out", help="Output CSV or JSON path."),
    format: str = typer.Option("csv", "--format", help="Export format: csv or json."),
    json: bool = typer.Option(False, "--json/--text", help="Output format (default: text)."),
) -> None:
    try:
        fmt = str(format).strip().lower()
        if fmt not in {"csv", "json"}:
            raise OpalError("--format must be one of: csv, json")
        payload = load_selection_batch(resolve_config_path(config), round_selector=round, run_id=run_id)
        output_path = Path(out)
        if output_path.exists() and output_path.is_dir():
            raise OpalError(f"--out must be a file, got directory: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            write_json(output_path, payload, indent=2)
        else:
            pd.DataFrame(payload["rows"]).to_csv(output_path, index=False)
        summary = {
            "schema_version": "opal.selection_batch_export.v1",
            "campaign": payload["campaign"],
            "as_of_round": payload["as_of_round"],
            "run_id": payload["run_id"],
            "output_path": str(output_path),
            "format": fmt,
            "row_count": payload["unique_count"],
        }
        if json:
            json_out(summary)
        else:
            print_stdout(f"wrote {summary['row_count']} batch rows to {output_path} ({fmt})")
    except OpalError as exc:
        if json:
            json_error("selection-batch export", exc)
        else:
            opal_error("selection-batch export", exc)
        raise typer.Exit(code=exc.exit_code)
    except Exception as exc:
        internal_error("selection-batch export", exc)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
