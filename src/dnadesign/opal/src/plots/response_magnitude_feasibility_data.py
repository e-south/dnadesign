"""Ledger contract for Response-Magnitude Feasibility decision plots."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from dnadesign.opal.api.response_magnitude_feasibility import (
    ResponseMagnitudeFeasibilityComponents,
    binary_target_mask,
    calibrate_response_magnitude_feasibility,
    score_response_magnitude_feasibility,
)

from ..analysis.ledger import read_labels, read_runs
from ..core.utils import ExitCodes, OpalError
from ._events_util import load_events, resolve_outputs_dir
from .sfxi_diag_data import labels_asof_round, resolve_run_id, resolve_single_round

OBJECTIVE_NAME = "response_magnitude_feasibility_v1"
FEASIBILITY_REF = "feasibility_margin"
RESPONSE_REF = "response_separation"
ON_MAGNITUDE_REF = "on_magnitude_floor"
OFF_MAGNITUDE_REF = "off_magnitude_ceiling"
REQUIRED_SCORE_REFS = (
    FEASIBILITY_REF,
    RESPONSE_REF,
    ON_MAGNITUDE_REF,
    OFF_MAGNITUDE_REF,
)


@dataclass(frozen=True)
class ResponseMagnitudeFeasibilityPlotData:
    """One unambiguous run of objective components and calibrated margins."""

    frame: pd.DataFrame
    observed_frame: pd.DataFrame
    calibration: dict[str, float]
    state_ids: tuple[str, ...]
    target_mask: tuple[int, ...]
    round_k: int
    run_id: str


def load_response_magnitude_feasibility_plot_data(context: Any) -> ResponseMagnitudeFeasibilityPlotData:
    """Load and verify one RMF prediction ledger."""

    outputs_dir = resolve_outputs_dir(context)
    runs = read_runs(outputs_dir / "ledger" / "runs.parquet")
    round_k = resolve_single_round(runs, round_selector=context.rounds)
    run_id = resolve_run_id(runs, round_k=round_k, run_id=context.run_id)
    if not run_id:
        raise OpalError("RMF plots require an explicit run_id in the run ledger.", ExitCodes.BAD_ARGS)
    run_row = _single_run_row(
        runs,
        round_k=round_k,
        run_id=run_id,
        selection_view_id=context.selection_view_id,
    )
    params = _objective_params(run_row)
    calibration = params.get("calibration")
    state_ids = tuple(str(value) for value in params["state_ids"])
    target_mask = tuple(int(value) for value in binary_target_mask(params.get("target_mask")).astype(int).tolist())
    if len(state_ids) != len(target_mask):
        raise OpalError("RMF state_ids and target_mask are misaligned.", ExitCodes.CONTRACT_VIOLATION)

    events = load_events(
        outputs_dir,
        {
            "as_of_round",
            "run_id",
            "id",
            "pred__score_channels",
            "view__rank_competition",
            "view__is_selected",
        },
        round_selector=[round_k],
        selection_view_id=context.selection_view_id,
        run_id=run_id,
    )
    frame = response_magnitude_feasibility_plot_frame(
        events,
        calibration=calibration,
        selection_view_id=context.selection_view_id,
    )
    labels = labels_asof_round(read_labels(outputs_dir / "ledger" / "labels.parquet"), round_k=round_k)
    observed_frame = response_magnitude_feasibility_observed_frame(
        labels.to_pandas(),
        target_mask=target_mask,
        calibration=calibration,
    )
    return ResponseMagnitudeFeasibilityPlotData(
        frame=frame,
        observed_frame=observed_frame,
        calibration={key: float(value) for key, value in calibration.items()},
        state_ids=state_ids,
        target_mask=target_mask,
        round_k=round_k,
        run_id=run_id,
    )


def response_magnitude_feasibility_observed_frame(
    labels: pd.DataFrame,
    *,
    target_mask: Sequence[int | float],
    calibration: Mapping[str, object],
) -> pd.DataFrame:
    """Reduce observed RMF labels to the same component space as predictions."""

    required = {"id", "observed_round", "y_obs"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise OpalError(f"RMF label ledger is missing columns: {missing}.", ExitCodes.CONTRACT_VIOLATION)
    if labels.empty:
        raise OpalError("RMF label ledger contains no rows for the selected round.", ExitCodes.CONTRACT_VIOLATION)
    duplicate_events = labels.duplicated(subset=["id", "observed_round"], keep=False)
    if duplicate_events.any():
        sample = labels.loc[duplicate_events, "id"].astype(str).tolist()[:5]
        raise OpalError(
            f"RMF label ledger contains duplicate id/round events (sample_ids={sample}).",
            ExitCodes.CONTRACT_VIOLATION,
        )
    current = labels.sort_values(["id", "observed_round"], kind="mergesort").drop_duplicates("id", keep="last")
    try:
        values = np.asarray([list(value) for value in current["y_obs"]], dtype=float)
        scored = score_response_magnitude_feasibility(
            values,
            target_mask=target_mask,
            calibration=calibration,
        )
    except (TypeError, ValueError) as exc:
        raise OpalError(
            f"RMF observed labels violate the eight-value objective contract: {exc}", ExitCodes.CONTRACT_VIOLATION
        ) from exc
    return pd.DataFrame(
        {
            "id": current["id"].astype(str).to_numpy(),
            RESPONSE_REF: scored.components.response_separation,
            ON_MAGNITUDE_REF: scored.components.on_magnitude_floor,
            OFF_MAGNITUDE_REF: scored.components.off_magnitude_ceiling,
            "response_constraint_margin": scored.response_constraint_margin,
            "on_magnitude_constraint_margin": scored.on_magnitude_constraint_margin,
            "off_magnitude_constraint_margin": scored.off_magnitude_constraint_margin,
            FEASIBILITY_REF: scored.feasibility_margin,
            "feasible": scored.feasibility_margin >= 0.0,
        }
    ).sort_values("id", kind="mergesort", ignore_index=True)


def response_magnitude_feasibility_plot_frame(
    events: pd.DataFrame,
    *,
    calibration: Mapping[str, object],
    selection_view_id: str,
) -> pd.DataFrame:
    """Parse score channels and verify the persisted maximin score."""

    required = {
        "as_of_round",
        "run_id",
        "id",
        "pred__score_channels",
        "view__rank_competition",
        "view__is_selected",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise OpalError(
            f"RMF prediction ledger is missing columns: {missing}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if events.empty:
        raise OpalError("RMF prediction ledger contains no rows.", ExitCodes.CONTRACT_VIOLATION)
    if events["id"].astype(str).duplicated().any():
        raise OpalError(
            "RMF prediction ledger contains duplicate candidate IDs.",
            ExitCodes.CONTRACT_VIOLATION,
        )

    channel_rows = [
        parse_response_magnitude_feasibility_channels(value, selection_view_id=selection_view_id)
        for value in events["pred__score_channels"]
    ]
    channels = pd.DataFrame.from_records(channel_rows, index=events.index)
    frame = pd.concat([events.drop(columns=["pred__score_channels"]).copy(), channels], axis=1)
    for column in ("view__rank_competition", *REQUIRED_SCORE_REFS):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    numeric = frame[["view__rank_competition", *REQUIRED_SCORE_REFS]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise OpalError("RMF prediction ledger contains non-finite values.", ExitCodes.CONTRACT_VIOLATION)
    if (frame["view__rank_competition"] < 1).any():
        raise OpalError("RMF selection ranks must be positive.", ExitCodes.CONTRACT_VIOLATION)
    if frame["view__is_selected"].isna().any():
        raise OpalError("RMF selected flags must be present.", ExitCodes.CONTRACT_VIOLATION)
    frame["view__is_selected"] = frame["view__is_selected"].astype(bool)

    components = ResponseMagnitudeFeasibilityComponents(
        response_separation=frame[RESPONSE_REF].to_numpy(dtype=float),
        on_magnitude_floor=frame[ON_MAGNITUDE_REF].to_numpy(dtype=float),
        off_magnitude_ceiling=frame[OFF_MAGNITUDE_REF].to_numpy(dtype=float),
    )
    scored = calibrate_response_magnitude_feasibility(components, calibration=calibration)
    persisted = frame[FEASIBILITY_REF].to_numpy(dtype=float)
    if not np.allclose(persisted, scored.feasibility_margin, rtol=1e-10, atol=1e-12):
        max_error = float(np.max(np.abs(persisted - scored.feasibility_margin)))
        raise OpalError(
            f"Persisted feasibility margins disagree with canonical objective math; max_abs_error={max_error:.3g}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    frame["response_constraint_margin"] = scored.response_constraint_margin
    frame["on_magnitude_constraint_margin"] = scored.on_magnitude_constraint_margin
    frame["off_magnitude_constraint_margin"] = scored.off_magnitude_constraint_margin
    frame["feasible"] = scored.feasibility_margin >= 0.0
    return frame.sort_values(["view__rank_competition", "id"], kind="mergesort").reset_index(drop=True)


def parse_response_magnitude_feasibility_channels(
    payload: object,
    *,
    selection_view_id: str,
) -> dict[str, float]:
    """Extract the four canonical channels from one ledger payload."""

    if isinstance(payload, np.ndarray):
        payload = payload.tolist()
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise OpalError("pred__score_channels must be a sequence of name/value mappings.", ExitCodes.CONTRACT_VIOLATION)
    parsed: dict[str, float] = {}
    for item in payload:
        if not isinstance(item, Mapping) or set(item) != {"name", "value"}:
            raise OpalError(
                "Each pred__score_channels entry must contain exactly name and value.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        name = str(item["name"]).strip()
        if not name:
            raise OpalError("Score-channel names must be non-empty.", ExitCodes.CONTRACT_VIOLATION)
        prefix = f"{selection_view_id}/"
        if not name.startswith(prefix):
            continue
        name = name.removeprefix(prefix)
        if name in parsed:
            raise OpalError(f"Duplicate score channel in prediction row: {name}.", ExitCodes.CONTRACT_VIOLATION)
        try:
            value = float(item["value"])
        except (TypeError, ValueError) as exc:
            raise OpalError(f"Score channel {name!r} must be numeric.", ExitCodes.CONTRACT_VIOLATION) from exc
        if not np.isfinite(value):
            raise OpalError(f"Score channel {name!r} must be finite.", ExitCodes.CONTRACT_VIOLATION)
        parsed[name] = value
    missing = sorted(set(REQUIRED_SCORE_REFS) - set(parsed))
    if missing:
        raise OpalError(
            f"RMF prediction row is missing score channels: {missing}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return {name: parsed[name] for name in REQUIRED_SCORE_REFS}


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
        raise OpalError(f"Run ledger is missing RMF fields: {missing}.", ExitCodes.CONTRACT_VIOLATION)
    rows = runs.filter((pl.col("as_of_round") == int(round_k)) & (pl.col("run_id") == str(run_id)))
    if rows.height != 1:
        raise OpalError(
            f"Expected one run row for round={round_k}, run_id={run_id!r}; found {rows.height}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    row = rows.to_dicts()[0]
    try:
        objective_defs = json.loads(str(row["objective__defs_json"]))
        selection_defs = json.loads(str(row["selection_views__defs_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OpalError(f"RMF run metadata is invalid JSON: {exc}", ExitCodes.CONTRACT_VIOLATION) from exc
    objective_matches = [item for item in objective_defs if item.get("selection_view_id") == selection_view_id]
    selection_matches = [item for item in selection_defs if item.get("selection_view_id") == selection_view_id]
    if len(objective_matches) != 1 or len(selection_matches) != 1:
        raise OpalError(
            f"RMF selection view {selection_view_id!r} is not defined exactly once.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    objective = objective_matches[0]
    selection = selection_matches[0]
    if str(objective.get("objective_name")) != OBJECTIVE_NAME:
        raise OpalError(
            f"RMF plots require objective {OBJECTIVE_NAME!r}; found {objective.get('objective_name')!r}.",
            ExitCodes.BAD_ARGS,
        )
    expected_score_ref = f"{selection_view_id}/{FEASIBILITY_REF}"
    if str(selection.get("score_ref")) != expected_score_ref:
        raise OpalError(
            f"RMF decision plots require selection score_ref {expected_score_ref!r}; "
            f"found {selection.get('score_ref')!r}.",
            ExitCodes.BAD_ARGS,
        )
    return {"objective_params": objective.get("params")}


def _objective_params(run_row: Mapping[str, object]) -> Mapping[str, object]:
    params = run_row.get("objective_params")
    if not isinstance(params, Mapping):
        raise OpalError("Run ledger RMF objective params must be a mapping.", ExitCodes.CONTRACT_VIOLATION)
    if set(params) != {"state_ids", "target_mask", "calibration"}:
        raise OpalError(
            "RMF objective params must contain exactly state_ids, target_mask, and calibration.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if not isinstance(params.get("calibration"), Mapping):
        raise OpalError("RMF calibration must be a mapping.", ExitCodes.CONTRACT_VIOLATION)
    return params


__all__ = [
    "FEASIBILITY_REF",
    "OBJECTIVE_NAME",
    "OFF_MAGNITUDE_REF",
    "ON_MAGNITUDE_REF",
    "RESPONSE_REF",
    "ResponseMagnitudeFeasibilityPlotData",
    "load_response_magnitude_feasibility_plot_data",
    "parse_response_magnitude_feasibility_channels",
    "response_magnitude_feasibility_observed_frame",
    "response_magnitude_feasibility_plot_frame",
]
