"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/ledger/predictions.py

Manifest-ledger prediction reads with explicit round and run contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from ...core.utils import ExitCodes, OpalError
from .io import scan_predictions
from .prediction_scope import (
    apply_round_filter as _apply_round_filter,
)
from .prediction_scope import (
    apply_row_filters as _apply_row_filters,
)
from .prediction_scope import (
    require_run_id_if_ambiguous as _require_run_id_if_ambiguous,
)
from .prediction_scope import (
    resolve_round_for_run_id as _resolve_round_for_run_id,
)
from .prediction_scope import (
    select_existing_columns as _select_existing_columns,
)
from .prediction_scope import (
    selected_rounds as _selected_rounds,
)
from .rounds import RoundSelector

_SELECTION_VIEW_COLUMNS = {
    "selection_view_id",
    "objective_name",
    "selection_name",
    "score_ref",
    "view__score",
    "view__score_ref",
    "view__selection_score",
    "view__rank_competition",
    "view__is_selected",
    "view__top_k",
    "view__uncertainty",
    "view__uncertainty_ref",
    "view__diagnostics",
}


def read_predictions(
    pred_dir: Path,
    *,
    columns: Sequence[str] | None = None,
    round_selector: RoundSelector | None = None,
    run_id: str | None = None,
    runs_df: pl.DataFrame | None = None,
    row_filters: Sequence[Mapping[str, Any]] | None = None,
    allow_missing: bool = False,
    require_run_id: bool = True,
) -> pl.DataFrame:
    lf = scan_predictions(pred_dir)
    if run_id is not None:
        if runs_df is None or runs_df.is_empty():
            raise OpalError(
                "run_id was provided but outputs/ledger/runs.parquet is missing or empty. "
                "Pass runs_df or call CampaignAnalysis.read_predictions so OPAL can resolve run_id -> as_of_round. "
                "Use `opal runs list` or `opal status --with-ledger` to find valid run_id values.",
                ExitCodes.BAD_ARGS,
            )
        run_round = _resolve_round_for_run_id(str(run_id), runs_df)
        if round_selector in (None, "unspecified", "latest"):
            round_selector = [run_round]
        elif round_selector != "all":
            selected = _selected_rounds(round_selector, runs_df)
            if run_round not in selected:
                raise OpalError(
                    f"run_id {run_id!r} belongs to as_of_round={run_round}, "
                    f"but round_selector={round_selector!r} excludes it.",
                    ExitCodes.BAD_ARGS,
                )
    _require_run_id_if_ambiguous(
        runs_df=runs_df,
        round_selector=round_selector,
        run_id=run_id,
        require_run_id=require_run_id,
    )
    want = _select_existing_columns(lf, columns, allow_missing=allow_missing)
    lf = _apply_round_filter(lf, round_selector=round_selector, runs_df=runs_df)
    if run_id is not None:
        lf = lf.filter(pl.col("run_id") == str(run_id))
    lf = _apply_row_filters(lf, row_filters)
    if want is not None:
        lf = lf.select(want)
    return lf.collect()


def read_selection_view_predictions(
    pred_dir: Path,
    *,
    selection_view_id: str,
    columns: Sequence[str] | None = None,
    round_selector: RoundSelector | None = None,
    run_id: str | None = None,
    runs_df: pl.DataFrame | None = None,
    row_filters: Sequence[Mapping[str, Any]] | None = None,
    allow_missing: bool = False,
    require_run_id: bool = True,
) -> pl.DataFrame:
    """Project one named selection view from the shared prediction ledger."""

    view_id = str(selection_view_id).strip()
    if not view_id:
        raise OpalError("selection_view_id must be non-empty.", ExitCodes.BAD_ARGS)
    requested_columns = None if columns is None else list(columns)
    shared_columns = (
        None
        if requested_columns is None
        else [column for column in requested_columns if column not in _SELECTION_VIEW_COLUMNS]
    )
    if shared_columns is not None:
        shared_columns.append("pred__selection_views")
    frame = read_predictions(
        pred_dir,
        columns=shared_columns,
        round_selector=round_selector,
        run_id=run_id,
        runs_df=runs_df,
        row_filters=None,
        allow_missing=allow_missing,
        require_run_id=require_run_id,
    )
    if frame.is_empty():
        return frame.drop("pred__selection_views")
    if "pred__selection_views" not in frame.columns:
        raise OpalError(
            "outputs/ledger/predictions is missing pred__selection_views.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    projected = (
        frame.explode("pred__selection_views")
        .unnest("pred__selection_views")
        .filter(pl.col("selection_view_id") == view_id)
    )
    if projected.is_empty():
        raise OpalError(
            f"selection view {view_id!r} is not present in the selected prediction rows.",
            ExitCodes.BAD_ARGS,
        )
    renamed = projected.rename(
        {
            "score": "view__score",
            "score_ref": "view__score_ref",
            "selection_score": "view__selection_score",
            "rank_competition": "view__rank_competition",
            "is_selected": "view__is_selected",
            "top_k": "view__top_k",
            "uncertainty": "view__uncertainty",
            "uncertainty_ref": "view__uncertainty_ref",
            "diagnostics": "view__diagnostics",
        }
    )
    projected = _apply_row_filters(renamed.lazy(), row_filters)
    want = _select_existing_columns(projected, requested_columns, allow_missing=allow_missing)
    if want is not None:
        projected = projected.select(want)
    return projected.collect()
