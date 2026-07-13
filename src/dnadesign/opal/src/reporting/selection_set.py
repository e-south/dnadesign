"""Public selection-view and logical selection-batch contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ..analysis.ledger import read_selection_view_predictions
from ..config.loader import load_config
from ..core.rounds import resolve_round_index_from_runs
from ..core.utils import OpalError
from ..storage.ledger import LedgerReader
from ..storage.workspace import CampaignWorkspace
from .summary import select_run_meta
from .verify_outputs import compare_selection_to_ledger, read_selection_artifact, read_selection_table

SELECTION_SET_SCHEMA_VERSION = "opal.selection_set.v2"
SELECTION_BATCH_SCHEMA_VERSION = "opal.selection_batch.v1"


def _campaign_json(cfg_path: Path, cfg: Any, ws: CampaignWorkspace) -> dict[str, str]:
    return {
        "slug": str(cfg.campaign.slug),
        "config_path": str(Path(cfg_path).resolve()),
        "workdir": str(ws.workdir),
    }


def _resolve_run(
    reader: LedgerReader,
    *,
    round_selector: str | int | None,
    run_id: str | None,
) -> tuple[pd.DataFrame, pd.Series]:
    runs = reader.read_runs()
    round_index = resolve_round_index_from_runs(
        runs,
        None if round_selector is None else str(round_selector),
        allow_none=True,
    )
    return runs, select_run_meta(runs, round_sel=round_index, run_id=run_id)


def _artifact_path(run_row: pd.Series, *, key: str, explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    artifacts = run_row.get("artifacts")
    if isinstance(artifacts, dict) and key in artifacts:
        value = artifacts[key]
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
            return Path(value[1])
        if isinstance(value, str):
            return Path(value)
    return None


def _require_view(cfg: Any, selection_view_id: str) -> str:
    view_id = str(selection_view_id).strip()
    configured = [view.id for view in cfg.selection_views]
    if view_id not in configured:
        raise OpalError(f"Unknown selection view {view_id!r}. Available: {configured}")
    return view_id


def _selected_rows(frame: pd.DataFrame, *, campaign_slug: str, run_id: str, as_of_round: int) -> list[dict[str, Any]]:
    required = {
        "id",
        "sequence",
        "view__is_selected",
        "view__rank_competition",
        "view__score",
        "view__selection_score",
        "run_id",
        "as_of_round",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise OpalError(f"Selection-view projection is missing required columns: {missing}")
    selected = frame.loc[frame["view__is_selected"].astype(bool)].copy()
    if selected.empty:
        raise OpalError(f"Selection view has no selected rows for campaign={campaign_slug}")
    if selected["id"].astype(str).duplicated().any():
        raise OpalError("Selection view contains duplicate selected IDs.")
    if selected["sequence"].isna().any() or selected["sequence"].astype(str).str.strip().eq("").any():
        raise OpalError("Selection view contains missing or blank sequences.")
    if not (selected["run_id"].astype(str) == str(run_id)).all():
        raise OpalError("Selection-view projection contains mixed run IDs.")
    if not (pd.to_numeric(selected["as_of_round"]).astype(int) == int(as_of_round)).all():
        raise OpalError("Selection-view projection contains mixed rounds.")
    selected["view__rank_competition"] = pd.to_numeric(selected["view__rank_competition"]).astype(int)
    selected = selected.sort_values(["view__rank_competition", "id"], kind="stable")
    return [
        {
            "id": str(row.id),
            "sequence": str(row.sequence),
            "selection_rank": rank,
            "rank_competition": int(row.view__rank_competition),
            "score": float(row.view__score),
            "selection_score": float(row.view__selection_score),
            "run_id": str(row.run_id),
            "as_of_round": int(row.as_of_round),
        }
        for rank, row in enumerate(selected.itertuples(index=False), start=1)
    ]


def load_selection_set(
    config_path: str | Path,
    *,
    selection_view_id: str,
    round_selector: str | int | None = None,
    run_id: str | None = None,
    selection_path: str | Path | None = None,
    verify_artifact: bool = True,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Load one named selection set from a shared OPAL round."""

    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path)
    view_id = _require_view(cfg, selection_view_id)
    ws = CampaignWorkspace.from_config(cfg, cfg_path)
    reader = LedgerReader(ws)
    runs, run_row = _resolve_run(reader, round_selector=round_selector, run_id=run_id)
    resolved_run_id = str(run_row["run_id"])
    as_of_round = int(run_row["as_of_round"])
    projected = read_selection_view_predictions(
        ws.ledger_predictions_dir,
        selection_view_id=view_id,
        columns=[
            "id",
            "sequence",
            "run_id",
            "as_of_round",
            "view__is_selected",
            "view__rank_competition",
            "view__score",
            "view__selection_score",
        ],
        round_selector=as_of_round,
        run_id=resolved_run_id,
        runs_df=pl.from_pandas(runs),
    ).to_pandas()
    rows = _selected_rows(
        projected,
        campaign_slug=str(cfg.campaign.slug),
        run_id=resolved_run_id,
        as_of_round=as_of_round,
    )
    resolved_path = _artifact_path(
        run_row,
        key="selection/selections.parquet",
        explicit=selection_path,
    )
    verification: dict[str, Any]
    if not verify_artifact:
        verification = {"status": "skipped", "reason": "verification disabled"}
    elif resolved_path is None:
        raise OpalError(
            "Run ledger is missing the selection/selections.parquet artifact reference. "
            "Use --selection-path only for an explicit audit override."
        )
    else:
        artifact = read_selection_table(resolved_path)
        missing = sorted({"id", "selection_view_id", "score", "selection_score"} - set(artifact.columns))
        if missing:
            raise OpalError(f"Selection data missing required columns: {missing}")
        artifact = artifact.loc[artifact["selection_view_id"].astype(str) == view_id].copy()
        if artifact.empty:
            raise OpalError(f"Selection artifact has no rows for selection view {view_id!r}.")
        if artifact["id"].isna().any() or artifact["id"].astype(str).str.strip().eq("").any():
            raise OpalError(f"Selection artifact has null or blank IDs for selection view {view_id!r}.")
        if artifact["id"].astype(str).duplicated().any():
            raise OpalError(f"Selection artifact has duplicate IDs for selection view {view_id!r}.")
        summary, mismatches = compare_selection_to_ledger(artifact, projected, eps=eps)
        verification = {
            "status": "pass" if int(summary["mismatch_count"]) == 0 else "fail",
            "selection_path": str(resolved_path),
            "summary": summary,
            "mismatch_count": int(summary["mismatch_count"]),
            "mismatches": mismatches.head(10).to_dict(orient="records"),
        }
    return {
        "schema_version": SELECTION_SET_SCHEMA_VERSION,
        "campaign": _campaign_json(cfg_path, cfg, ws),
        "selection_view_id": view_id,
        "round_selector": None if round_selector is None else str(round_selector),
        "as_of_round": as_of_round,
        "run_id": resolved_run_id,
        "selected_count": len(rows),
        "selection_path": None if resolved_path is None else str(resolved_path),
        "verification": verification,
        "rows": rows,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_selection_batch(
    config_path: str | Path,
    *,
    round_selector: str | int | None = None,
    run_id: str | None = None,
    selection_batch_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the deduplicated logical union of all selection sets for one run."""

    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path)
    ws = CampaignWorkspace.from_config(cfg, cfg_path)
    reader = LedgerReader(ws)
    _, run_row = _resolve_run(reader, round_selector=round_selector, run_id=run_id)
    resolved_run_id = str(run_row["run_id"])
    as_of_round = int(run_row["as_of_round"])
    path = _artifact_path(
        run_row,
        key="selection/selection_batch.parquet",
        explicit=selection_batch_path,
    )
    if path is None:
        raise OpalError("Run ledger is missing the selection/selection_batch.parquet artifact reference.")
    frame = read_selection_artifact(path, required_columns=("id", "selection_view_ids", "selection_memberships"))
    if "selection_batch_key" not in frame.columns or "deduplicate_by" not in frame.columns:
        raise OpalError("Selection batch is missing selection_batch_key or deduplicate_by.")
    if frame["selection_batch_key"].astype(str).duplicated().any():
        raise OpalError("Selection batch contains duplicate selection_batch_key values.")
    rows = [{key: _json_value(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]
    return {
        "schema_version": SELECTION_BATCH_SCHEMA_VERSION,
        "campaign": _campaign_json(cfg_path, cfg, ws),
        "round_selector": None if round_selector is None else str(round_selector),
        "as_of_round": as_of_round,
        "run_id": resolved_run_id,
        "selection_batch_path": str(path),
        "unique_count": len(rows),
        "rows": rows,
    }
