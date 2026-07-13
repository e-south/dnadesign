"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/verify_outputs.py

CLI for validating selection outputs against ledger predictions. Resolves.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl
import typer

from ...analysis.ledger import read_selection_view_predictions
from ...core.rounds import resolve_round_index_from_runs
from ...core.utils import ExitCodes, OpalError, print_stdout
from ...reporting.summary import select_run_meta
from ...reporting.verify_outputs import (
    compare_selection_to_ledger,
    read_selection_table,
    resolve_selection_path_from_artifacts,
)
from ...storage.ledger import LedgerReader
from ...storage.workspace import CampaignWorkspace
from ..guidance_hints import maybe_print_hints
from ..registry import cli_command
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
    print_config_context,
    resolve_config_path,
)


@cli_command(
    "verify-outputs",
    help="Compare selection artifacts against ledger predictions for a run.",
)
def verify_outputs(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    view: str = typer.Option(..., "--view", help="Selection view ID to verify."),
    round: Optional[str] = typer.Option(None, "--round", "-r", help="Round selector: int or 'latest'."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Explicit run_id to compare."),
    selection_path: Optional[Path] = typer.Option(
        None,
        "--selection-path",
        help="Explicit selections.parquet audit override (default: run-ledger artifact reference).",
    ),
    eps: float = typer.Option(1e-6, "--eps", help="Mismatch tolerance for numeric comparisons."),
    no_hints: bool = typer.Option(False, "--no-hints", help="Disable next-step hints in text output."),
    json: bool = typer.Option(False, "--json/--text", help="Output format (default: text)."),
) -> None:
    try:
        if round and run_id:
            raise OpalError("Provide only one of --round or --run-id.")
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        ws = CampaignWorkspace.from_config(cfg, cfg_path)
        reader = LedgerReader(ws)
        runs_df = reader.read_runs()
        round_sel = resolve_round_index_from_runs(runs_df, round, allow_none=True) if run_id is None else None
        run_row = select_run_meta(runs_df, round_sel=round_sel, run_id=run_id)
        run_id = str(run_row.get("run_id"))
        as_of_round = int(run_row.get("as_of_round"))

        artifacts = run_row.get("artifacts")
        sel_path = selection_path or resolve_selection_path_from_artifacts(artifacts, run_id=run_id)
        if sel_path is None:
            raise OpalError(
                "Run ledger is missing the selection/selections.parquet artifact reference. "
                "Use --selection-path only for an explicit audit override."
            )

        selection_df = read_selection_table(Path(sel_path))
        if "selection_view_id" not in selection_df.columns:
            raise OpalError("Selection data missing selection_view_id.")
        selection_df = selection_df.loc[selection_df["selection_view_id"].astype(str) == str(view)].copy()
        if selection_df.empty:
            raise OpalError(f"Selection artifact contains no rows for selection view {view!r}.")
        ledger_df = read_selection_view_predictions(
            ws.ledger_predictions_dir,
            selection_view_id=view,
            columns=["id"],
            round_selector=as_of_round,
            run_id=run_id,
            runs_df=pl.from_pandas(runs_df),
        ).to_pandas()
        summary, mismatches = compare_selection_to_ledger(selection_df, ledger_df, eps=eps)
        summary.update(
            {
                "run_id": run_id,
                "as_of_round": as_of_round,
                "selection_view_id": view,
                "selection_path": str(sel_path),
                "ledger_rows": int(ledger_df.shape[0]),
            }
        )

        if json:
            json_out(
                {
                    "summary": summary,
                    "mismatches": mismatches.head(10).to_dict(orient="records"),
                }
            )
        else:
            print_config_context(cfg_path, cfg=cfg)
            print_stdout("verify-outputs")
            print_stdout(
                f"- run_id: {summary['run_id']}  round: {summary['as_of_round']}  "
                f"view: {summary['selection_view_id']}  "
                f"selection: {summary['selection_path']}"
            )
            print_stdout(
                f"- compared: {summary['rows_compared']}  mismatches: {summary['mismatch_count']}  "
                f"max_abs_diff: {summary['max_abs_diff']}"
            )
            if summary["mismatch_count"] > 0:
                print_stdout("- top mismatches:")
                print_stdout(mismatches.head(10).to_string(index=False))
            maybe_print_hints(
                command_name="verify-outputs",
                cfg_path=cfg_path,
                no_hints=no_hints,
                json_output=json,
            )
    except OpalError as e:
        opal_error("verify-outputs", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("verify-outputs", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
