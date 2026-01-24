# ABOUTME: CLI for validating selection outputs against ledger predictions.
# ABOUTME: Resolves selection artifacts and reports mismatches for a run.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/verify_outputs.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

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
from ..registry import cli_command
from ._common import internal_error, json_out, load_cli_config, opal_error, print_config_context, resolve_config_path


@cli_command("verify-outputs", help="Compare selection artifacts against ledger predictions for a run.")
def verify_outputs(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: Optional[str] = typer.Option(None, "--round", "-r", help="Round selector: int or 'latest'."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Explicit run_id to compare."),
    selection_path: Optional[Path] = typer.Option(
        None,
        "--selection-path",
        help="Optional selection_top_k.csv/.parquet path (defaults to run artifacts).",
    ),
    eps: float = typer.Option(1e-6, "--eps", help="Mismatch tolerance for numeric comparisons."),
    json: bool = typer.Option(False, "--json/--human", help="Output format (default: human)."),
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
            round_dir = ws.round_dir(as_of_round)
            candidate = round_dir / "selection" / "selection_top_k.parquet"
            if not candidate.exists():
                candidate = round_dir / "selection" / "selection_top_k.csv"
            if candidate.exists():
                sel_path = candidate
        if sel_path is None:
            raise OpalError("Could not resolve selection output path; provide --selection-path explicitly.")

        selection_df = read_selection_table(Path(sel_path))
        ledger_df = reader.read_predictions(
            columns=["id", "pred__y_obj_scalar"],
            round_selector=as_of_round,
            run_id=run_id,
        )
        summary, mismatches = compare_selection_to_ledger(selection_df, ledger_df, eps=eps)
        summary.update(
            {
                "run_id": run_id,
                "as_of_round": as_of_round,
                "selection_path": str(sel_path),
                "ledger_rows": int(ledger_df.shape[0]),
            }
        )

        if json:
            json_out({"summary": summary, "mismatches": mismatches.head(10).to_dict(orient="records")})
        else:
            print_config_context(cfg_path, cfg=cfg)
            print_stdout("verify-outputs")
            print_stdout(
                f"- run_id: {summary['run_id']}  round: {summary['as_of_round']}  "
                f"selection: {summary['selection_path']}"
            )
            print_stdout(
                f"- compared: {summary['rows_compared']}  mismatches: {summary['mismatch_count']}  "
                f"max_abs_diff: {summary['max_abs_diff']}"
            )
            if summary["mismatch_count"] > 0:
                print_stdout("- top mismatches:")
                print_stdout(mismatches.head(10).to_string(index=False))
    except OpalError as e:
        opal_error("verify-outputs", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("verify-outputs", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
