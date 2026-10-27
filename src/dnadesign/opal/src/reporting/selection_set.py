"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/reporting/selection_set.py

Public selection-view and logical selection-batch contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

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
from .selection_batch_contract import SELECTION_BATCH_REQUIRED_COLUMNS, validate_selection_batch_rows
from .selection_batch_verification import verify_file_digest, verify_selection_batch_memberships
from .summary import select_run_meta
from .verify_outputs import compare_selection_to_ledger, read_selection_table

SELECTION_SET_SCHEMA_VERSION = "opal.selection_set.v2"
SELECTION_BATCH_SCHEMA_VERSION = "opal.selection_batch.v3"


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


def _artifact_reference(run_row: pd.Series, *, key: str) -> tuple[str, Path] | None:
    artifacts = run_row.get("artifacts")
    if not isinstance(artifacts, dict) or key not in artifacts:
        return None
    value = artifacts[key]
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    return str(value[0]), Path(value[1])


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


def load_selection_batch(
    config_path: str | Path,
    *,
    round_selector: str | int | None = None,
    run_id: str | None = None,
    selection_batch_path: str | Path | None = None,
    usr_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load the final deduplicated selection batch for one run."""

    cfg_path = Path(config_path).resolve()
    cfg = load_config(cfg_path, usr_root=usr_root)
    ws = CampaignWorkspace.from_config(cfg, cfg_path)
    reader = LedgerReader(ws)
    _, run_row = _resolve_run(reader, round_selector=round_selector, run_id=run_id)
    resolved_run_id = str(run_row["run_id"])
    as_of_round = int(run_row["as_of_round"])
    batch_reference = _artifact_reference(run_row, key="selection/selection_batch.parquet")
    if batch_reference is None:
        raise OpalError("Run ledger is missing the selection/selection_batch.parquet artifact reference.")
    batch_sha256, ledger_batch_path = batch_reference
    path = Path(selection_batch_path) if selection_batch_path is not None else ledger_batch_path
    if selection_batch_path is None:
        verified_batch_sha256 = verify_file_digest(
            path,
            expected_sha256=batch_sha256,
            artifact_key="selection/selection_batch.parquet",
        )
        batch_digest_status = "pass"
    else:
        verified_batch_sha256 = None
        batch_digest_status = "explicit_override"

    selections_reference = _artifact_reference(run_row, key="selection/selections.parquet")
    if selections_reference is None:
        raise OpalError("Run ledger is missing the selection/selections.parquet artifact reference.")
    selections_sha256, selections_path = selections_reference
    verified_selections_sha256 = verify_file_digest(
        selections_path,
        expected_sha256=selections_sha256,
        artifact_key="selection/selections.parquet",
    )
    frame = read_selection_table(path)
    missing = sorted(SELECTION_BATCH_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise OpalError(f"Selection batch is missing required columns: {missing}.")
    configured_view_ids = tuple(view.id for view in cfg.selection_views)
    deduplicate_by = str(cfg.selection_batch.deduplicate_by or "id").strip()
    allocation = cfg.selection_batch.allocation
    allocation_strategy = None if allocation is None else str(allocation.strategy)
    allocation_view_priority = configured_view_ids if allocation is None else tuple(allocation.view_priority)
    quota_by_view = {view.id: int(view.selection.params["top_k"]) for view in cfg.selection_views}
    allocation_trace_path: Path | None = None
    allocation_trace_digest: dict[str, str | None] = {
        "status": "not_applicable",
        "expected_sha256": None,
        "verified_sha256": None,
    }
    if allocation is not None:
        trace_reference = _artifact_reference(run_row, key="selection/allocation_trace.parquet")
        if trace_reference is None:
            raise OpalError("Run ledger is missing the selection/allocation_trace.parquet artifact reference.")
        trace_sha256, allocation_trace_path = trace_reference
        verified_trace_sha256 = verify_file_digest(
            allocation_trace_path,
            expected_sha256=trace_sha256,
            artifact_key="selection/allocation_trace.parquet",
        )
        allocation_trace_digest = {
            "status": "pass",
            "expected_sha256": trace_sha256,
            "verified_sha256": verified_trace_sha256,
        }
    rows = validate_selection_batch_rows(
        frame,
        campaign_slug=str(cfg.campaign.slug),
        run_id=resolved_run_id,
        as_of_round=as_of_round,
        configured_view_ids=configured_view_ids,
        deduplicate_by=deduplicate_by,
        allocation_strategy=allocation_strategy,
        allocation_view_priority=allocation_view_priority,
        quota_by_view=quota_by_view,
    )
    membership_verification = verify_selection_batch_memberships(
        rows,
        selections_path=selections_path,
        allocation_trace_path=allocation_trace_path,
        campaign_slug=str(cfg.campaign.slug),
        run_id=resolved_run_id,
        as_of_round=as_of_round,
        deduplicate_by=deduplicate_by,
    )
    return {
        "schema_version": SELECTION_BATCH_SCHEMA_VERSION,
        "campaign": _campaign_json(cfg_path, cfg, ws),
        "round_selector": None if round_selector is None else str(round_selector),
        "as_of_round": as_of_round,
        "run_id": resolved_run_id,
        "selection_batch_path": str(path),
        "deduplicate_by": deduplicate_by,
        "allocation_strategy": allocation_strategy or "logical_union",
        "unique_count": len(rows),
        "verification": {
            "status": "pass",
            "batch_digest": {
                "status": batch_digest_status,
                "expected_sha256": batch_sha256,
                "verified_sha256": verified_batch_sha256,
            },
            "selections_digest": {
                "status": "pass",
                "expected_sha256": selections_sha256,
                "verified_sha256": verified_selections_sha256,
            },
            "allocation_trace_digest": allocation_trace_digest,
            "memberships": membership_verification,
        },
        "rows": rows,
    }
