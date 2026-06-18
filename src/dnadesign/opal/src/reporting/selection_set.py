"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/reporting/selection_set.py

Public selection-set contract for OPAL run outputs. Downstream study handoffs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..config.loader import load_config
from ..core.rounds import resolve_round_index_from_runs
from ..core.utils import OpalError
from ..storage.ledger import LedgerReader
from ..storage.workspace import CampaignWorkspace
from .summary import select_run_meta
from .verify_outputs import (
    compare_selection_to_ledger,
    read_selection_artifact,
    resolve_selection_path_from_artifacts,
)

SELECTION_SET_SCHEMA_VERSION = "opal.selection_set.v1"


def _campaign_json(cfg_path: Path, cfg: Any, ws: CampaignWorkspace) -> dict[str, str]:
    return {
        "slug": str(cfg.campaign.slug),
        "config_path": str(Path(cfg_path).resolve()),
        "workdir": str(ws.workdir),
    }


def _bool_series(values: pd.Series, *, column: str) -> pd.Series:
    parsed: list[bool] = []
    for value in values.tolist():
        if isinstance(value, bool):
            parsed.append(value)
            continue
        if isinstance(value, int | float) and not pd.isna(value):
            if int(value) in {0, 1} and float(value) == float(int(value)):
                parsed.append(bool(int(value)))
                continue
        text = str(value).strip().lower()
        if text in {"true", "1"}:
            parsed.append(True)
            continue
        if text in {"false", "0"}:
            parsed.append(False)
            continue
        raise OpalError(f"{column} must be boolean-like, got {value!r}")
    return pd.Series(parsed, index=values.index)


def _prediction_columns(reader: LedgerReader) -> list[str]:
    available = set(reader.predictions_schema_columns())
    required = [
        "id",
        "sequence",
        "sel__is_selected",
        "sel__rank_competition",
        "run_id",
        "as_of_round",
    ]
    missing = sorted(set(required).difference(available))
    if missing:
        raise OpalError(f"OPAL prediction ledger missing required selection-set columns: {missing}")
    optional = ["pred__score_selected"]
    return [column for column in [*required, *optional] if column in available]


def _selected_rows(
    predictions: pd.DataFrame, *, campaign_slug: str, as_of_round: int, run_id: str
) -> list[dict[str, Any]]:
    required = {
        "id",
        "sequence",
        "sel__is_selected",
        "sel__rank_competition",
        "run_id",
        "as_of_round",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise OpalError(f"OPAL prediction ledger missing required selection-set columns: {missing}")

    frame = predictions.copy()
    frame["id"] = frame["id"].astype(str).str.strip()
    if frame["id"].eq("").any():
        raise OpalError(f"OPAL prediction ledger contains blank IDs for campaign={campaign_slug}")
    frame["sequence"] = frame["sequence"].astype(str).str.strip()
    if frame["sequence"].eq("").any():
        raise OpalError(f"OPAL prediction ledger contains blank sequences for campaign={campaign_slug}")
    frame["run_id"] = frame["run_id"].astype(str)
    frame["as_of_round"] = pd.to_numeric(frame["as_of_round"], errors="raise").astype(int)
    if not (frame["run_id"] == str(run_id)).all():
        raise OpalError(f"OPAL prediction ledger contains mixed run_id values for campaign={campaign_slug}")
    if not (frame["as_of_round"] == int(as_of_round)).all():
        raise OpalError(f"OPAL prediction ledger contains mixed as_of_round values for campaign={campaign_slug}")

    selected_mask = _bool_series(frame["sel__is_selected"], column="sel__is_selected")
    selected = frame.loc[selected_mask].copy()
    if selected.empty:
        raise OpalError(f"OPAL selection set has no selected rows for campaign={campaign_slug}")
    if selected["id"].duplicated().any():
        dup_ids = sorted(selected.loc[selected["id"].duplicated(), "id"].astype(str).unique().tolist())
        raise OpalError(f"OPAL selection set contains duplicate selected IDs: {dup_ids[:10]}")

    selected["sel__rank_competition"] = pd.to_numeric(
        selected["sel__rank_competition"],
        errors="raise",
    ).astype(int)
    if (selected["sel__rank_competition"] <= 0).any():
        raise OpalError(f"OPAL selected rows contain non-positive competition ranks for campaign={campaign_slug}")
    selected = selected.sort_values(["sel__rank_competition", "id"], kind="mergesort").reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for selection_rank, row in enumerate(selected.to_dict(orient="records"), start=1):
        out = {
            "id": str(row["id"]),
            "sequence": str(row["sequence"]),
            "selection_rank": int(selection_rank),
            "sel__rank_competition": int(row["sel__rank_competition"]),
            "run_id": str(row["run_id"]),
            "as_of_round": int(row["as_of_round"]),
        }
        if "pred__score_selected" in row and not pd.isna(row["pred__score_selected"]):
            out["pred__score_selected"] = float(row["pred__score_selected"])
        rows.append(out)
    return rows


def _selection_path_for_run(
    *,
    run_row: pd.Series,
    run_id: str,
    as_of_round: int,
    ws: CampaignWorkspace,
    explicit_selection_path: Path | None,
) -> Path | None:
    if explicit_selection_path is not None:
        return Path(explicit_selection_path)
    artifacts = run_row.get("artifacts")
    resolved = resolve_selection_path_from_artifacts(artifacts, run_id=run_id)
    if resolved is not None:
        return resolved
    fallback = ws.round_dir(as_of_round) / "selection" / "selection_top_k.csv"
    if fallback.exists():
        return fallback
    return None


def _verify_selection_artifact(
    *,
    reader: LedgerReader,
    as_of_round: int,
    run_id: str,
    selection_path: Path | None,
    eps: float,
    verify_artifact: bool,
) -> dict[str, Any]:
    if not verify_artifact:
        return {"status": "skipped", "reason": "verification disabled"}
    if selection_path is None:
        return {"status": "not_checked", "reason": "selection artifact not found"}

    selection_df = read_selection_artifact(Path(selection_path), required_columns=("id", "pred__score_selected"))
    ledger_df = reader.read_predictions(
        columns=["id", "pred__score_selected"],
        round_selector=as_of_round,
        run_id=run_id,
    )
    summary, mismatches = compare_selection_to_ledger(selection_df, ledger_df, eps=eps)
    status = "pass" if int(summary["mismatch_count"]) == 0 else "fail"
    return {
        "status": status,
        "selection_path": str(selection_path),
        "summary": summary,
        "mismatch_count": int(summary["mismatch_count"]),
        "mismatches": mismatches.head(10).to_dict(orient="records"),
    }


def load_selection_set(
    config_path: str | Path,
    *,
    round_selector: str | int | None = None,
    run_id: str | None = None,
    selection_path: str | Path | None = None,
    verify_artifact: bool = True,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Load the selected rows for one unambiguous OPAL run."""

    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path)
    ws = CampaignWorkspace.from_config(cfg, cfg_path)
    reader = LedgerReader(ws)

    runs_df = reader.read_runs()
    round_sel = resolve_round_index_from_runs(
        runs_df,
        None if round_selector is None else str(round_selector),
        allow_none=True,
    )
    run_row = select_run_meta(runs_df, round_sel=round_sel, run_id=run_id)
    resolved_run_id = str(run_row.get("run_id"))
    as_of_round = int(run_row.get("as_of_round"))
    predictions = reader.read_predictions(
        columns=_prediction_columns(reader),
        round_selector=as_of_round,
        run_id=resolved_run_id,
    )
    rows = _selected_rows(
        predictions,
        campaign_slug=str(cfg.campaign.slug),
        as_of_round=as_of_round,
        run_id=resolved_run_id,
    )

    resolved_selection_path = _selection_path_for_run(
        run_row=run_row,
        run_id=resolved_run_id,
        as_of_round=as_of_round,
        ws=ws,
        explicit_selection_path=(Path(selection_path) if selection_path is not None else None),
    )
    verification = _verify_selection_artifact(
        reader=reader,
        as_of_round=as_of_round,
        run_id=resolved_run_id,
        selection_path=resolved_selection_path,
        eps=float(eps),
        verify_artifact=bool(verify_artifact),
    )

    return {
        "schema_version": SELECTION_SET_SCHEMA_VERSION,
        "campaign": _campaign_json(cfg_path, cfg, ws),
        "round_selector": None if round_selector is None else str(round_selector),
        "as_of_round": as_of_round,
        "run_id": resolved_run_id,
        "selected_count": int(len(rows)),
        "selection_path": None if resolved_selection_path is None else str(resolved_selection_path),
        "verification": verification,
        "rows": rows,
    }
