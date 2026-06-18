"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/selection_set.py

CLI for OPAL selection-set inspection and export.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from ...core.utils import ExitCodes, OpalError, print_stdout, write_json
from ...reporting.selection_set import load_selection_set
from ..registry import cli_group
from ._common import internal_error, json_error, json_out, opal_error, resolve_config_path

selection_set_app = typer.Typer(no_args_is_help=True, help="Inspect and export selected OPAL rows.")
cli_group("selection-set", help="Inspect and export selected OPAL rows.")(selection_set_app)


@selection_set_app.command("show", help="Show the selected rows for one OPAL run.")
def selection_set_show(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: Optional[str] = typer.Option(None, "--round", "-r", help="Round selector: int or 'latest'."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Explicit run_id to show."),
    selection_path: Optional[Path] = typer.Option(
        None,
        "--selection-path",
        help="Optional selection artifact path used for verification.",
    ),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Verify selection artifact when available."),
    json: bool = typer.Option(False, "--json/--text", help="Output format (default: text)."),
) -> None:
    try:
        payload = load_selection_set(
            resolve_config_path(config),
            round_selector=round,
            run_id=run_id,
            selection_path=selection_path,
            verify_artifact=verify,
        )
        if json:
            json_out(payload)
        else:
            print_stdout("selection-set")
            print_stdout(
                f"- campaign: {payload['campaign']['slug']}  round: {payload['as_of_round']}  "
                f"run_id: {payload['run_id']}"
            )
            print_stdout(
                f"- selected: {payload['selected_count']}  verification: {payload['verification'].get('status')}"
            )
            print_stdout(pd.DataFrame(payload["rows"]).to_string(index=False))
    except OpalError as e:
        if json:
            json_error("selection-set show", e)
        else:
            opal_error("selection-set show", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("selection-set show", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@selection_set_app.command("export", help="Export the selected rows for one OPAL run.")
def selection_set_export(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: Optional[str] = typer.Option(None, "--round", "-r", help="Round selector: int or 'latest'."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Explicit run_id to export."),
    out: Path = typer.Option(..., "--out", help="Output CSV or JSON path."),
    format: str = typer.Option("csv", "--format", help="Export format: csv or json."),
    selection_path: Optional[Path] = typer.Option(
        None,
        "--selection-path",
        help="Optional selection artifact path used for verification.",
    ),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Verify selection artifact when available."),
    json: bool = typer.Option(False, "--json/--text", help="Output format (default: text)."),
) -> None:
    try:
        fmt = str(format).strip().lower()
        if fmt not in {"csv", "json"}:
            raise OpalError("--format must be one of: csv, json")
        payload = load_selection_set(
            resolve_config_path(config),
            round_selector=round,
            run_id=run_id,
            selection_path=selection_path,
            verify_artifact=verify,
        )
        output_path = Path(out)
        if output_path.exists() and output_path.is_dir():
            raise OpalError(f"--out must be a file, got directory: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            write_json(output_path, payload, indent=2)
        else:
            pd.DataFrame(payload["rows"]).to_csv(output_path, index=False)
        summary = {
            "schema_version": "opal.selection_set_export.v1",
            "campaign": payload["campaign"],
            "as_of_round": payload["as_of_round"],
            "run_id": payload["run_id"],
            "output_path": str(output_path),
            "format": fmt,
            "row_count": int(payload["selected_count"]),
            "verification": payload["verification"],
        }
        if json:
            json_out(summary)
        else:
            print_stdout(
                f"wrote {summary['row_count']} selection rows to {summary['output_path']} ({summary['format']})"
            )
    except OpalError as e:
        if json:
            json_error("selection-set export", e)
        else:
            opal_error("selection-set export", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("selection-set export", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
