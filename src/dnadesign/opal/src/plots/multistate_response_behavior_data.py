"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/multistate_response_behavior_data.py

Ledger contract for Multistate Response Behavior decision plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl

from dnadesign.opal.api.multistate_response_behavior import (
    binary_target_mask,
    score_multistate_response_behavior,
    validated_softmin_scale,
    validated_state_ids,
)

from ..analysis.ledger import read_run_observed_events, read_runs
from ..core.utils import ExitCodes, OpalError
from ._events_util import load_events, resolve_outputs_dir
from ._ledger_contracts import validated_competition_ranks, validated_selected_flags
from ._run_resolution import parse_run_view_definitions, resolve_run_id, resolve_single_round

OBJECTIVE_NAME = "multistate_response_behavior_v1"
BEHAVIOR_SCORE_REF = "behavior_score"
HARD_BOTTLENECK_REF = "hard_bottleneck_clearance"
RESPONSE_FAMILY_REF = "response_family_score"
ON_SIGNAL_FAMILY_REF = "on_signal_family_score"
OFF_SIGNAL_SUPPRESSION_FAMILY_REF = "off_signal_suppression_family_score"
SUMMARY_DETAIL_SCOPE = "summary"
SELECTED_COORDINATE_DETAIL_SCOPE = "selected_coordinates"
PlotDetailScope = Literal["summary", "selected_coordinates"]


@dataclass(frozen=True)
class MultistateResponseBehaviorPlotData:
    """One unambiguous run of verified behavior predictions and observations."""

    frame: pd.DataFrame
    observed_frame: pd.DataFrame
    state_ids: tuple[str, ...]
    target_mask: tuple[int, ...]
    softmin_scale: float
    coordinate_labels: tuple[str, ...]
    round_k: int
    run_id: str
    selected_coordinate_frame: pd.DataFrame | None = None


def load_multistate_response_behavior_plot_data(
    context: Any,
    *,
    detail_scope: PlotDetailScope = SUMMARY_DETAIL_SCOPE,
) -> MultistateResponseBehaviorPlotData:
    """Load one behavior-objective run with explicitly scoped coordinate detail."""

    if detail_scope not in {SUMMARY_DETAIL_SCOPE, SELECTED_COORDINATE_DETAIL_SCOPE}:
        raise OpalError(
            "Behavior plot detail_scope must be 'summary' or 'selected_coordinates'.",
            ExitCodes.BAD_ARGS,
        )

    outputs_dir = resolve_outputs_dir(context)
    runs = read_runs(outputs_dir / "ledger" / "runs.parquet")
    round_k = resolve_single_round(runs, round_selector=context.rounds)
    run_id = resolve_run_id(runs, round_k=round_k, run_id=context.run_id)
    if not run_id:
        raise OpalError("Behavior plots require an explicit run_id in the run ledger.", ExitCodes.BAD_ARGS)
    run_row = _single_run_row(
        runs,
        round_k=round_k,
        run_id=run_id,
        selection_view_id=context.selection_view_id,
    )
    params = _objective_params(run_row)
    try:
        state_ids = validated_state_ids(params["state_ids"])
        target_mask = tuple(int(value) for value in binary_target_mask(params["target_mask"]).astype(int).tolist())
        softmin_scale = validated_softmin_scale(params["softmin_scale"])
    except (TypeError, ValueError) as exc:
        raise OpalError(
            f"Behavior run parameters violate the public objective contract: {exc}",
            ExitCodes.CONTRACT_VIOLATION,
        ) from exc

    events = load_events(
        outputs_dir,
        {
            "as_of_round",
            "run_id",
            "id",
            "pred__y_hat_model",
            "pred__score_channels",
            "view__rank_competition",
            "view__is_selected",
        },
        round_selector=[round_k],
        selection_view_id=context.selection_view_id,
        run_id=run_id,
    )
    frame = multistate_response_behavior_plot_frame(
        events,
        state_ids=state_ids,
        target_mask=target_mask,
        softmin_scale=softmin_scale,
        selection_view_id=context.selection_view_id,
    )
    observed_snapshot = read_run_observed_events(
        runs,
        outputs_dir=outputs_dir,
        round_k=round_k,
        run_id=run_id,
    )
    context.data_paths["run_observed_events_parquet"] = observed_snapshot.path
    observed_frame = multistate_response_behavior_observed_frame(
        observed_snapshot.frame.to_pandas(),
        state_ids=state_ids,
        target_mask=target_mask,
        softmin_scale=softmin_scale,
    )
    coordinate_labels = _coordinate_labels(
        state_ids=state_ids,
        target_mask=target_mask,
        softmin_scale=softmin_scale,
    )
    selected_coordinate_frame = None
    if detail_scope == SELECTED_COORDINATE_DETAIL_SCOPE:
        selected_coordinate_frame = _selected_coordinate_frame(
            frame,
            state_ids=state_ids,
            target_mask=target_mask,
            softmin_scale=softmin_scale,
            coordinate_labels=coordinate_labels,
        )
    frame = frame.drop(columns=["pred__y_hat_model"])
    return MultistateResponseBehaviorPlotData(
        frame=frame,
        observed_frame=observed_frame,
        state_ids=state_ids,
        target_mask=target_mask,
        softmin_scale=softmin_scale,
        coordinate_labels=coordinate_labels,
        round_k=round_k,
        run_id=run_id,
        selected_coordinate_frame=selected_coordinate_frame,
    )


def multistate_response_behavior_plot_frame(
    events: pd.DataFrame,
    *,
    state_ids: Sequence[str],
    target_mask: Sequence[int | float],
    softmin_scale: object,
    selection_view_id: str,
) -> pd.DataFrame:
    """Replay behavior predictions and reject persisted score drift."""

    required = {
        "as_of_round",
        "run_id",
        "id",
        "pred__y_hat_model",
        "pred__score_channels",
        "view__rank_competition",
        "view__is_selected",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise OpalError(
            f"Behavior prediction ledger is missing columns: {missing}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if events.empty:
        raise OpalError("Behavior prediction ledger contains no rows.", ExitCodes.CONTRACT_VIOLATION)
    if events["id"].astype(str).duplicated().any():
        raise OpalError("Behavior prediction ledger contains duplicate candidate IDs.", ExitCodes.CONTRACT_VIOLATION)

    frame = events.copy()
    frame[BEHAVIOR_SCORE_REF] = [
        _parse_behavior_score_channel(payload, selection_view_id=selection_view_id)
        for payload in frame.pop("pred__score_channels")
    ]
    frame["view__rank_competition"] = validated_competition_ranks(
        frame["view__rank_competition"],
        objective_label="Behavior",
    )
    frame[BEHAVIOR_SCORE_REF] = pd.to_numeric(frame[BEHAVIOR_SCORE_REF], errors="raise")
    if not np.isfinite(frame[BEHAVIOR_SCORE_REF].to_numpy(dtype=float)).all():
        raise OpalError("Behavior prediction ledger contains non-finite values.", ExitCodes.CONTRACT_VIOLATION)
    frame["view__is_selected"] = validated_selected_flags(
        frame["view__is_selected"],
        objective_label="Behavior",
    )

    values = _vector_matrix(frame["pred__y_hat_model"], context="behavior prediction vectors")
    scored = _score(
        values,
        state_ids=state_ids,
        target_mask=target_mask,
        softmin_scale=softmin_scale,
        context="Behavior prediction vectors",
    )
    persisted = frame[BEHAVIOR_SCORE_REF].to_numpy(dtype=float)
    if not np.allclose(persisted, scored.behavior_score, rtol=1e-10, atol=1e-12):
        max_error = float(np.max(np.abs(persisted - scored.behavior_score)))
        raise OpalError(
            f"Persisted behavior scores disagree with canonical objective math; max_abs_error={max_error:.3g}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    _attach_scored_columns(frame, scored)
    return frame.sort_values(["view__rank_competition", "id"], kind="mergesort").reset_index(drop=True)


def multistate_response_behavior_observed_frame(
    labels: pd.DataFrame,
    *,
    state_ids: Sequence[str],
    target_mask: Sequence[int | float],
    softmin_scale: object,
) -> pd.DataFrame:
    """Replay every run-pinned observed event under one behavior target view."""

    required = {"id", "observed_round", "batch_id", "display_label", "y_obs"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise OpalError(f"Behavior label ledger is missing columns: {missing}.", ExitCodes.CONTRACT_VIOLATION)
    if labels.empty:
        raise OpalError("Behavior label ledger contains no rows for the selected run.", ExitCodes.CONTRACT_VIOLATION)
    duplicate_events = labels.duplicated(subset=["id", "observed_round", "batch_id"], keep=False)
    if duplicate_events.any():
        sample = labels.loc[duplicate_events, "id"].astype(str).tolist()[:5]
        raise OpalError(
            f"Behavior label ledger contains duplicate candidate/round/batch events (sample_ids={sample}).",
            ExitCodes.CONTRACT_VIOLATION,
        )
    current = labels.copy()
    current["batch_key"] = [
        str(batch_id) if not pd.isna(batch_id) else f"round-{int(observed_round)}"
        for observed_round, batch_id in current[["observed_round", "batch_id"]].itertuples(index=False, name=None)
    ]
    current = current.sort_values(["observed_round", "batch_key", "id"], kind="mergesort").reset_index(drop=True)
    values = _vector_matrix(current["y_obs"], context="behavior observed vectors")
    scored = _score(
        values,
        state_ids=state_ids,
        target_mask=target_mask,
        softmin_scale=softmin_scale,
        context="Behavior observed vectors",
    )
    frame = pd.DataFrame(
        {
            "id": current["id"].astype(str).to_numpy(),
            "observed_round": current["observed_round"].astype(int).to_numpy(),
            "batch_id": current["batch_id"].astype("string").to_numpy(),
            "batch_key": current["batch_key"].astype(str).to_numpy(),
            "display_label": current["display_label"].astype("string").to_numpy(),
        }
    )
    _attach_scored_columns(frame, scored)
    return frame.sort_values(["observed_round", "batch_key", "id"], kind="mergesort", ignore_index=True)


def _score(
    values: np.ndarray,
    *,
    state_ids: Sequence[str],
    target_mask: Sequence[int | float],
    softmin_scale: object,
    context: str,
):
    try:
        return score_multistate_response_behavior(
            values,
            state_ids=state_ids,
            target_mask=target_mask,
            softmin_scale=softmin_scale,
        )
    except (TypeError, ValueError) as exc:
        raise OpalError(f"{context} violate the objective contract: {exc}", ExitCodes.CONTRACT_VIOLATION) from exc


def _attach_scored_columns(
    frame: pd.DataFrame,
    scored: Any,
    *,
    include_coordinate_detail: bool = False,
) -> None:
    frame[BEHAVIOR_SCORE_REF] = np.asarray(scored.behavior_score, dtype=float)
    frame[HARD_BOTTLENECK_REF] = np.asarray(scored.hard_bottleneck_clearance, dtype=float)
    frame[RESPONSE_FAMILY_REF] = np.asarray(scored.response_family_score, dtype=float)
    frame[ON_SIGNAL_FAMILY_REF] = np.asarray(scored.on_signal_family_score, dtype=float)
    frame[OFF_SIGNAL_SUPPRESSION_FAMILY_REF] = np.asarray(scored.off_signal_suppression_family_score, dtype=float)
    if include_coordinate_detail:
        frame["coordinate_clearances"] = [row.tolist() for row in scored.coordinate_clearances]
        frame["coordinate_weights"] = [row.tolist() for row in scored.coordinate_weights]
        frame["coordinate_labels"] = [list(scored.coordinate_labels) for _ in range(len(frame))]
    limiting_indices = np.asarray(scored.limiting_coordinate_index, dtype=int)
    frame["limiting_coordinate_index"] = limiting_indices
    frame["limiting_coordinate_label"] = pd.Categorical.from_codes(
        limiting_indices,
        categories=list(scored.coordinate_labels),
    )
    frame["all_reference_directions_met"] = np.asarray(scored.all_reference_directions_met, dtype=bool)


def _vector_matrix(values: pd.Series, *, context: str) -> np.ndarray:
    try:
        matrix = np.asarray(values.tolist(), dtype=float)
    except (TypeError, ValueError) as exc:
        raise OpalError(f"{context} must be finite numeric vectors.", ExitCodes.CONTRACT_VIOLATION) from exc
    return matrix


def _parse_behavior_score_channel(payload: object, *, selection_view_id: str) -> float:
    if isinstance(payload, np.ndarray):
        payload = payload.tolist()
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise OpalError("pred__score_channels must be a sequence of name/value mappings.", ExitCodes.CONTRACT_VIOLATION)
    prefix = f"{selection_view_id}/"
    parsed: dict[str, float] = {}
    for item in payload:
        if not isinstance(item, Mapping) or set(item) != {"name", "value"}:
            raise OpalError(
                "Each pred__score_channels entry must contain exactly name and value.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        name = str(item["name"]).strip()
        if not name.startswith(prefix):
            continue
        channel = name.removeprefix(prefix)
        if channel in parsed:
            raise OpalError(f"Duplicate behavior score channel: {channel}.", ExitCodes.CONTRACT_VIOLATION)
        try:
            value = float(item["value"])
        except (TypeError, ValueError) as exc:
            raise OpalError(
                f"Behavior score channel {channel!r} must be numeric.", ExitCodes.CONTRACT_VIOLATION
            ) from exc
        if not np.isfinite(value):
            raise OpalError(f"Behavior score channel {channel!r} must be finite.", ExitCodes.CONTRACT_VIOLATION)
        parsed[channel] = value
    if set(parsed) != {BEHAVIOR_SCORE_REF}:
        raise OpalError(
            "Behavior prediction row must contain exactly the active view's behavior_score channel; "
            f"found={sorted(parsed)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return parsed[BEHAVIOR_SCORE_REF]


def _coordinate_labels(
    *,
    state_ids: Sequence[str],
    target_mask: Sequence[int | float],
    softmin_scale: object,
) -> tuple[str, ...]:
    state_count = len(state_ids)
    scored = _score(
        np.zeros((1, 2 * state_count), dtype=float),
        state_ids=state_ids,
        target_mask=target_mask,
        softmin_scale=softmin_scale,
        context="Behavior coordinate contract",
    )
    return tuple(str(value) for value in scored.coordinate_labels)


def _selected_coordinate_frame(
    frame: pd.DataFrame,
    *,
    state_ids: Sequence[str],
    target_mask: Sequence[int | float],
    softmin_scale: object,
    coordinate_labels: Sequence[str],
) -> pd.DataFrame:
    selected = frame.loc[frame["view__is_selected"]].copy()
    if selected.empty:
        return selected
    scored = _score(
        _vector_matrix(selected["pred__y_hat_model"], context="selected behavior prediction vectors"),
        state_ids=state_ids,
        target_mask=target_mask,
        softmin_scale=softmin_scale,
        context="Selected behavior prediction vectors",
    )
    scored_labels = tuple(str(value) for value in scored.coordinate_labels)
    if scored_labels != tuple(coordinate_labels):
        raise OpalError("Behavior coordinate labels drifted within one run.", ExitCodes.CONTRACT_VIOLATION)
    _attach_scored_columns(selected, scored, include_coordinate_detail=True)
    return selected.drop(columns=["pred__y_hat_model"]).reset_index(drop=True)


def _single_run_row(
    runs: pl.DataFrame,
    *,
    round_k: int,
    run_id: str,
    selection_view_id: str,
) -> dict[str, object]:
    required = {"as_of_round", "run_id", "objective__defs_json", "selection_views__defs_json"}
    missing = sorted(required - set(runs.columns))
    if missing:
        raise OpalError(f"Run ledger is missing behavior fields: {missing}.", ExitCodes.CONTRACT_VIOLATION)
    rows = runs.filter((pl.col("as_of_round") == int(round_k)) & (pl.col("run_id") == str(run_id)))
    if rows.height != 1:
        raise OpalError(
            f"Expected one behavior run row for round={round_k}, run_id={run_id!r}; found {rows.height}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    row = rows.to_dicts()[0]
    objective_defs = parse_run_view_definitions(
        row["objective__defs_json"],
        field_label="Behavior run objective definitions",
    )
    selection_defs = parse_run_view_definitions(
        row["selection_views__defs_json"],
        field_label="Behavior run selection-view definitions",
    )
    objective_matches = [item for item in objective_defs if item.get("selection_view_id") == selection_view_id]
    selection_matches = [item for item in selection_defs if item.get("selection_view_id") == selection_view_id]
    if len(objective_matches) != 1 or len(selection_matches) != 1:
        raise OpalError(
            f"Behavior selection view {selection_view_id!r} is not defined exactly once.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    objective = objective_matches[0]
    selection = selection_matches[0]
    if str(objective.get("objective_name")) != OBJECTIVE_NAME:
        raise OpalError(
            f"Behavior plots require objective {OBJECTIVE_NAME!r}; found {objective.get('objective_name')!r}.",
            ExitCodes.BAD_ARGS,
        )
    expected_score_ref = f"{selection_view_id}/{BEHAVIOR_SCORE_REF}"
    if str(selection.get("score_ref")) != expected_score_ref:
        raise OpalError(
            f"Behavior plots require selection score_ref {expected_score_ref!r}; found {selection.get('score_ref')!r}.",
            ExitCodes.BAD_ARGS,
        )
    return {"objective_params": objective.get("params")}


def _objective_params(run_row: Mapping[str, object]) -> Mapping[str, object]:
    params = run_row.get("objective_params")
    if not isinstance(params, Mapping):
        raise OpalError("Run ledger behavior objective params must be a mapping.", ExitCodes.CONTRACT_VIOLATION)
    required = {"state_ids", "target_mask", "softmin_scale"}
    if set(params) != required:
        raise OpalError(
            "Behavior objective params must contain exactly state_ids, target_mask, and softmin_scale.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return params


__all__ = [
    "BEHAVIOR_SCORE_REF",
    "HARD_BOTTLENECK_REF",
    "OBJECTIVE_NAME",
    "OFF_SIGNAL_SUPPRESSION_FAMILY_REF",
    "ON_SIGNAL_FAMILY_REF",
    "PlotDetailScope",
    "RESPONSE_FAMILY_REF",
    "SELECTED_COORDINATE_DETAIL_SCOPE",
    "SUMMARY_DETAIL_SCOPE",
    "MultistateResponseBehaviorPlotData",
    "load_multistate_response_behavior_plot_data",
    "multistate_response_behavior_observed_frame",
    "multistate_response_behavior_plot_frame",
]
