"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/objective_meta.py

Inspect one selection view's objective metadata and prediction diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import polars as pl
import typer

from ...analysis.ledger import read_selection_view_predictions
from ...core.config_resolve import resolve_campaign_config_path
from ...core.rounds import resolve_round_index_from_runs
from ...core.utils import ExitCodes, OpalError, print_stdout
from ...reporting.summary import select_run_meta
from ...storage.ledger import LedgerReader
from ...storage.workspace import CampaignWorkspace
from ..registry import cli_command
from ._common import internal_error, json_out, load_cli_config, opal_error, print_config_context


def _definitions(raw: Any, *, field: str) -> list[dict[str, Any]]:
    try:
        value = raw if isinstance(raw, list) else json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OpalError(f"Run metadata field {field} is invalid JSON: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise OpalError(f"Run metadata field {field} must contain a list of objects.")
    return [dict(item) for item in value]


def _view_definition(run_row: pd.Series, *, view_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    objectives = _definitions(run_row["objective__defs_json"], field="objective__defs_json")
    selections = _definitions(run_row["selection_views__defs_json"], field="selection_views__defs_json")
    objective_matches = [row for row in objectives if row.get("selection_view_id") == view_id]
    selection_matches = [row for row in selections if row.get("selection_view_id") == view_id]
    if len(objective_matches) != 1 or len(selection_matches) != 1:
        raise OpalError(f"Selection view {view_id!r} is not defined exactly once in run metadata.")
    return objective_matches[0], selection_matches[0]


def _diagnostic_keys(frame: pd.DataFrame) -> list[str]:
    keys: set[str] = set()
    for diagnostics in frame["view__diagnostics"].tolist():
        for item in diagnostics or []:
            if isinstance(item, dict) and item.get("name"):
                keys.add(str(item["name"]))
    return sorted(keys)


def _profile(frame: pd.DataFrame, diagnostic_keys: list[str]) -> dict[str, Any]:
    expanded = frame.copy()
    for key in diagnostic_keys:
        expanded[f"diagnostic/{key}"] = [
            next(
                (float(item["value"]) for item in (items or []) if item.get("name") == key),
                np.nan,
            )
            for items in expanded["view__diagnostics"]
        ]
    columns = [
        "view__score",
        "view__selection_score",
        "view__rank_competition",
        "view__uncertainty",
        *[f"diagnostic/{key}" for key in diagnostic_keys],
    ]
    rows = []
    for column in columns:
        if column not in expanded.columns:
            continue
        values = pd.to_numeric(expanded[column], errors="coerce")
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "column": column,
                "count": int(values.notna().sum()),
                "finite_count": int(finite.size),
                "min": None if finite.empty else float(finite.min()),
                "median": None if finite.empty else float(finite.median()),
                "max": None if finite.empty else float(finite.max()),
            }
        )
    return {"columns": rows}


@cli_command("objective-meta", help="Inspect objective metadata for one selection view and run.")
def cmd_objective_meta(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="campaign.yaml or directory",
        envvar="OPAL_CONFIG",
    ),
    view: str = typer.Option(..., "--view", help="Selection view ID."),
    round: Optional[str] = typer.Option(None, "--round", "-r", help="Round selector: integer or 'latest'."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Explicit run_id to inspect."),
    json_output: bool = typer.Option(False, "--json/--text", help="Output as JSON."),
    profile: bool = typer.Option(False, "--profile/--no-profile", help="Profile numeric view fields."),
) -> None:
    try:
        if round and run_id:
            raise OpalError("Provide only one of --round or --run-id.")
        cfg_path = resolve_campaign_config_path(config, allow_dir=True)
        cfg = load_cli_config(cfg_path)
        configured = [selection_view.id for selection_view in cfg.selection_views]
        if view not in configured:
            raise OpalError(f"Unknown selection view {view!r}. Available: {configured}")
        ws = CampaignWorkspace.from_config(cfg, cfg_path)
        reader = LedgerReader(ws)
        runs = reader.read_runs(
            columns=[
                "run_id",
                "as_of_round",
                "objective__defs_json",
                "selection_views__defs_json",
            ]
        )
        round_index = resolve_round_index_from_runs(runs, round, allow_none=True) if run_id is None else None
        run_row = select_run_meta(runs, round_sel=round_index, run_id=run_id)
        resolved_run_id = str(run_row["run_id"])
        as_of_round = int(run_row["as_of_round"])
        objective, selection = _view_definition(run_row, view_id=view)
        frame = read_selection_view_predictions(
            ws.ledger_predictions_dir,
            selection_view_id=view,
            round_selector=as_of_round,
            run_id=resolved_run_id,
            runs_df=pl.from_pandas(runs),
        ).to_pandas()
        diagnostic_keys = _diagnostic_keys(frame)
        out = {
            "schema_version": "opal.objective_meta.v2",
            "round": as_of_round,
            "run_id": resolved_run_id,
            "selection_view_id": view,
            "objective": objective,
            "selection": selection,
            "diagnostic_keys": diagnostic_keys,
        }
        if profile:
            out["profile"] = _profile(frame, diagnostic_keys)
        if json_output:
            json_out(out)
            return
        print_config_context(cfg_path, cfg=cfg)
        print_stdout(f"Round: {as_of_round}  run_id: {resolved_run_id}  selection view: {view}")
        print_stdout(f"Objective: {objective.get('objective_name')}")
        print_stdout("Diagnostics: " + (", ".join(diagnostic_keys) or "(none)"))
    except OpalError as exc:
        opal_error("objective-meta", exc)
        raise typer.Exit(code=exc.exit_code)
    except Exception as exc:
        internal_error("objective-meta", exc)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
