"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/record_show.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from ...core.rounds import resolve_round_index_from_state
from ...core.utils import ExitCodes, OpalError, print_stdout
from ...reporting.record_show import build_record_report
from ...storage.ledger import LedgerReader
from ...storage.workspace import CampaignWorkspace
from ..formatting import render_record_report_human
from ..registry import cli_command
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
    print_config_context,
    resolve_config_path,
    store_from_cfg,
)


def _resolve_run_id_alias(*, run_id: str | None, ledger_reader: LedgerReader) -> str | None:
    if run_id is None:
        return None
    rid = str(run_id).strip()
    if rid.lower() != "latest":
        return rid
    runs_df = ledger_reader.read_runs(columns=["run_id", "as_of_round"])
    if runs_df.empty:
        raise OpalError("outputs/ledger/runs.parquet is empty; cannot resolve --run-id latest.")
    if "run_id" not in runs_df.columns or "as_of_round" not in runs_df.columns:
        raise OpalError("outputs/ledger/runs.parquet missing required columns run_id/as_of_round.")
    runs_df = runs_df.copy()
    runs_df["as_of_round"] = runs_df["as_of_round"].astype(int)
    runs_df["run_id"] = runs_df["run_id"].astype(str)
    runs_df = runs_df[runs_df["run_id"].str.len() > 0]
    if runs_df.empty:
        raise OpalError("outputs/ledger/runs.parquet has no valid run_id rows.")
    row = runs_df.sort_values(["as_of_round", "run_id"]).tail(1).iloc[0]
    return str(row["run_id"])


def _resolve_id_from_selection_rank(
    *,
    ws: CampaignWorkspace,
    round_sel: str,
    selected_rank: int,
) -> str:
    round_index = resolve_round_index_from_state(ws.state_path, round_sel)
    sel_path = ws.round_selection_dir(round_index) / "selection_top_k.csv"
    if not sel_path.exists():
        raise OpalError(f"Selection CSV not found for round {round_index}: {sel_path}")

    sel_df = pd.read_csv(sel_path)
    required = {"id", "sel__rank_competition"}
    missing = sorted(required - set(sel_df.columns))
    if missing:
        raise OpalError(f"Selection CSV is missing required column(s) {missing}: {sel_path}")

    ranks = pd.to_numeric(sel_df["sel__rank_competition"], errors="coerce")
    if ranks.isna().any():
        raise OpalError(f"Selection CSV contains non-numeric sel__rank_competition values: {sel_path}")
    matches = sel_df.loc[ranks.astype(int) == int(selected_rank)]
    if matches.empty:
        raise OpalError(f"No selected record found at competition rank {selected_rank} in round {round_index}.")
    if len(matches) > 1:
        raise OpalError(f"Multiple records share competition rank {selected_rank} in round {round_index}; use --id.")

    resolved = str(matches.iloc[0]["id"]).strip()
    if not resolved:
        raise OpalError(f"Resolved id at competition rank {selected_rank} is empty in selection CSV: {sel_path}")
    return resolved


@cli_command(
    "record-show",
    help="Per-record report: ground truth & history; per-round predictions/ranks.",
)
def cmd_record_show(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    key: str = typer.Argument(None, help="ID or sequence (positional). Use --id/--sequence to disambiguate."),
    id: str = typer.Option(None, "--id"),
    sequence: str = typer.Option(None, "--sequence"),
    selected_rank: int = typer.Option(
        None,
        "--selected-rank",
        min=1,
        help="Resolve id from selection_top_k.csv by competition rank (1-based).",
    ),
    round: str = typer.Option(
        "latest",
        "--round",
        help="Round selector used with --selected-rank (int or 'latest').",
    ),
    run_id: str = typer.Option(None, "--run-id", help="Explicit run_id for ledger predictions (or 'latest')."),
    with_sequence: bool = typer.Option(True, "--with-sequence/--no-sequence"),
    json: bool = typer.Option(False, "--json"),
):
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        store = store_from_cfg(cfg)
        df = store.load()
        if not json:
            print_config_context(cfg_path, cfg=cfg, records_path=store.records_path)
        ws = CampaignWorkspace.from_config(cfg, cfg_path)
        ledger_reader = LedgerReader(ws)

        if selected_rank is not None:
            if id is not None or sequence is not None or key:
                raise OpalError("--selected-rank cannot be combined with key/--id/--sequence.")
            id = _resolve_id_from_selection_rank(
                ws=ws,
                round_sel=round,
                selected_rank=int(selected_rank),
            )

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

        run_id = _resolve_run_id_alias(run_id=run_id, ledger_reader=ledger_reader)

        report = build_record_report(
            df,
            cfg.campaign.slug,
            id_=id,
            sequence=sequence,
            with_sequence=with_sequence,
            ledger_reader=ledger_reader,
            records_path=store.records_path,
            run_id=run_id,
        )
        if "error" in report:
            raise OpalError(str(report["error"]))
        if json:
            json_out(report)
        else:
            print_stdout(render_record_report_human(report))
    except OpalError as e:
        opal_error("record-show", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("record-show", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
