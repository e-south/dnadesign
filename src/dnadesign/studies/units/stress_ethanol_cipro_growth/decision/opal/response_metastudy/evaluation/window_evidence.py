"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/window_evidence.py

Equal-footing assay evidence for declared Reader response windows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from dnadesign.opal import response_magnitude_feasibility_components

from .response_magnitude import RESPONSE_SEMANTICS

_VALUE_COLUMNS = tuple(f"{prefix}{state}" for prefix in ("r", "b") for state in ("00", "10", "01", "11"))
_EVENT_COLUMNS = tuple(f"{column}_event_half_range" for column in _VALUE_COLUMNS)
_MODEL_IDS = ("campaign_random_forest", "pls4", "pls6")


def build_response_window_evidence(
    *,
    labels: pd.DataFrame,
    margin_rows: pd.DataFrame,
    reader_designs: pd.DataFrame,
    reader_wells: pd.DataFrame,
    reader_traces: pd.DataFrame,
    model_screen: pd.DataFrame,
    reference_design_id: str,
    response_controls: Mapping[str, str],
) -> pd.DataFrame:
    """Compare every declared reduction without recomputing Reader-owned Y."""

    reduction_metadata = _reduction_metadata(labels)
    reduction_ids = tuple(reduction_metadata.index.astype(str))
    _assert_equal_reduction_universe(
        reduction_ids,
        margin_rows=margin_rows,
        reader_designs=reader_designs,
        reader_wells=reader_wells,
        model_screen=model_screen,
    )
    frames = (
        _anchor_evidence(reader_wells, reference_design_id=reference_design_id),
        _control_evidence(reader_designs, response_controls=response_controls),
        _growth_evidence(reader_traces, reduction_metadata=reduction_metadata),
        _event_evidence(labels),
        _repeat_evidence(reader_designs),
        _model_evidence(model_screen, reduction_ids=reduction_ids),
        _censoring_evidence(reader_designs, reader_wells, reader_traces, reduction_ids=reduction_ids),
    )
    result = reduction_metadata.reset_index()
    for frame in frames:
        result = result.merge(frame, on="reduction_id", how="left", validate="one_to_one")
    if result.isna().all(axis=0).any():
        empty_columns = result.columns[result.isna().all(axis=0)].tolist()
        raise ValueError(f"response-window evidence produced empty comparison columns: {empty_columns}")
    result["response_semantics"] = RESPONSE_SEMANTICS
    result["window_selection_basis"] = "assay_evidence_not_model_performance"
    result["model_evidence_use"] = "diagnostic_only"
    result["trajectory_role"] = "diagnostic_only_not_label_reduction"
    return result.sort_values("reduction_id", kind="mergesort").reset_index(drop=True)


def _reduction_metadata(labels: pd.DataFrame) -> pd.DataFrame:
    required = {
        "reduction_id",
        "reduction_method",
        "response_basis",
        "screen_role",
        "window_start_event_h",
        "window_end_event_h",
    }
    _require_columns(labels, required, context="response-window labels")
    fields = sorted(required - {"reduction_id"})
    cardinality = labels.groupby("reduction_id", sort=True)[fields].nunique(dropna=False)
    if cardinality.empty or cardinality.gt(1).any().any():
        raise ValueError("response-window reduction metadata must be complete and constant within each reduction.")
    return labels.groupby("reduction_id", sort=True)[fields].first()


def _assert_equal_reduction_universe(
    reduction_ids: tuple[str, ...],
    *,
    margin_rows: pd.DataFrame,
    reader_designs: pd.DataFrame,
    reader_wells: pd.DataFrame,
    model_screen: pd.DataFrame,
) -> None:
    expected = set(reduction_ids)
    if not expected:
        raise ValueError("response-window comparison requires at least one reduction.")
    for context, frame in (
        ("RMF component", margin_rows),
        ("Reader design", reader_designs),
        ("Reader well", reader_wells),
    ):
        _require_columns(frame, {"reduction_id"}, context=context)
        observed = set(frame["reduction_id"].astype(str))
        if observed != expected:
            raise ValueError(
                f"equal-footing {context} reductions disagree: missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}."
            )
    _require_columns(model_screen, {"representation_id", "model_id"}, context="model screen")
    identity = model_screen.loc[model_screen["representation_id"].astype(str).isin(expected)].copy()
    observed = set(identity["representation_id"].astype(str))
    if observed != expected:
        raise ValueError(
            "equal-footing model screen reductions disagree: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}."
        )
    model_sets = identity.groupby("representation_id")["model_id"].agg(lambda values: frozenset(map(str, values)))
    if model_sets.nunique() != 1:
        raise ValueError("equal-footing model screen requires the same fixed model set for every reduction.")


def _anchor_evidence(wells: pd.DataFrame, *, reference_design_id: str) -> pd.DataFrame:
    required = {
        "experiment_id",
        "design_id",
        "reduction_id",
        "state",
        "response_well",
        "magnitude_well",
        "is_reference",
    }
    _require_columns(wells, required, context="Reader wells")
    reference = wells.loc[
        wells["is_reference"].astype(bool) & wells["design_id"].astype(str).eq(str(reference_design_id))
    ].copy()
    if reference.empty:
        raise ValueError(f"Reader wells lack reference design {reference_design_id!r}.")
    group_keys = ["reduction_id", "experiment_id", "state"]
    within = reference.groupby(group_keys, sort=True).agg(
        response_range=("response_well", lambda values: float(np.ptp(values.to_numpy(dtype=float)))),
        magnitude_range=("magnitude_well", lambda values: float(np.ptp(values.to_numpy(dtype=float)))),
        response_median=("response_well", "median"),
        magnitude_median=("magnitude_well", "median"),
    )
    cross = (
        within.reset_index()
        .groupby(["reduction_id", "state"], sort=True)
        .agg(
            response_cross_experiment_range=(
                "response_median",
                lambda values: float(np.ptp(values.to_numpy(dtype=float))),
            ),
            magnitude_cross_experiment_range=(
                "magnitude_median",
                lambda values: float(np.ptp(values.to_numpy(dtype=float))),
            ),
            experiment_count=("experiment_id", "nunique"),
        )
    )
    records: list[dict[str, object]] = []
    for reduction_id, frame in within.groupby(level="reduction_id", sort=True):
        cross_frame = cross.loc[str(reduction_id)]
        records.append(
            {
                "reduction_id": str(reduction_id),
                "pdual_anchor_experiment_count": int(cross_frame["experiment_count"].max()),
                "pdual_response_within_experiment_median_range": float(frame["response_range"].median()),
                "pdual_response_within_experiment_max_range": float(frame["response_range"].max()),
                "pdual_magnitude_within_experiment_median_range": float(frame["magnitude_range"].median()),
                "pdual_magnitude_within_experiment_max_range": float(frame["magnitude_range"].max()),
                "pdual_response_cross_experiment_max_state_range": float(
                    cross_frame["response_cross_experiment_range"].max()
                ),
                "pdual_magnitude_cross_experiment_max_state_range": float(
                    cross_frame["magnitude_cross_experiment_range"].max()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _control_evidence(designs: pd.DataFrame, *, response_controls: Mapping[str, str]) -> pd.DataFrame:
    required = {"experiment_id", "reduction_id", "design_id", *_VALUE_COLUMNS}
    _require_columns(designs, required, context="Reader designs")
    if dict(response_controls) != {
        "ethanol": "pDual-10-spyp",
        "ciprofloxacin": "pDual-10-sulAp",
    }:
        raise ValueError("response controls must declare the study's exact SpyP and sulAp target views.")
    target_masks = {
        "ethanol": (0.0, 1.0, 0.0, 1.0),
        "ciprofloxacin": (0.0, 0.0, 1.0, 1.0),
    }
    result: pd.DataFrame | None = None
    for view_id, design_id in response_controls.items():
        rows = designs.loc[designs["design_id"].astype(str).eq(str(design_id))].copy()
        if rows.duplicated(subset=["reduction_id", "experiment_id"]).any():
            raise ValueError(f"response control {design_id!r} has duplicate experiment/reduction rows.")
        values = rows.loc[:, _VALUE_COLUMNS].to_numpy(dtype=float)
        components = response_magnitude_feasibility_components(values, target_mask=target_masks[view_id])
        rows["response_separation"] = components.response_separation
        prefix = "spyp_ethanol" if view_id == "ethanol" else "sulap_ciprofloxacin"
        rows = (
            rows.groupby("reduction_id", sort=True)
            .agg(
                **{
                    f"{prefix}_experiment_count": ("experiment_id", "nunique"),
                    f"{prefix}_response_separation_min": ("response_separation", "min"),
                    f"{prefix}_response_separation_median": ("response_separation", "median"),
                    f"{prefix}_response_separation_max": ("response_separation", "max"),
                }
            )
            .reset_index()
        )
        result = rows if result is None else result.merge(rows, on="reduction_id", how="outer", validate="one_to_one")
    if result is None or result.isna().any().any():
        raise ValueError("response controls are incomplete across declared reductions.")
    return result


def _growth_evidence(traces: pd.DataFrame, *, reduction_metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"experiment_id", "design_id", "position", "state", "time_from_event_h", "value", "signal_kind"}
    _require_columns(traces, required, context="Reader traces")
    growth = traces.loc[traces["signal_kind"].astype(str).eq("growth")].copy()
    if growth.empty:
        raise ValueError("Reader traces contain no growth observations.")
    records: list[dict[str, object]] = []
    trace_keys = ["experiment_id", "design_id", "position", "state"]
    for reduction_id, metadata in reduction_metadata.iterrows():
        start = float(metadata["window_start_event_h"])
        end = float(metadata["window_end_event_h"])
        window = growth.loc[growth["time_from_event_h"].between(start, end, inclusive="both")].copy()
        if window.empty:
            raise ValueError(f"growth traces have no observations for reduction {reduction_id!r}.")
        endpoint = (
            window.sort_values([*trace_keys, "time_from_event_h"], kind="mergesort")
            .groupby(trace_keys, sort=True, as_index=False)
            .tail(1)["value"]
            .to_numpy(dtype=float)
        )
        if endpoint.size == 0 or not np.isfinite(endpoint).all():
            raise ValueError(f"growth endpoint evidence is invalid for reduction {reduction_id!r}.")
        records.append(
            {
                "reduction_id": str(reduction_id),
                "growth_endpoint_well_count": int(endpoint.size),
                "growth_endpoint_od600_median": float(np.median(endpoint)),
                "growth_endpoint_od600_q90": float(np.quantile(endpoint, 0.90)),
                "growth_endpoint_od600_max": float(np.max(endpoint)),
            }
        )
    return pd.DataFrame.from_records(records)


def _event_evidence(labels: pd.DataFrame) -> pd.DataFrame:
    _require_columns(labels, {"reduction_id", *_EVENT_COLUMNS}, context="Reader labels")
    records: list[dict[str, object]] = []
    for reduction_id, frame in labels.groupby("reduction_id", sort=True):
        values = frame.loc[:, _EVENT_COLUMNS].to_numpy(dtype=float).ravel()
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"event sensitivity is invalid for reduction {reduction_id!r}.")
        records.append(
            {
                "reduction_id": str(reduction_id),
                "event_sensitivity_median_half_range": float(np.median(values)),
                "event_sensitivity_max_half_range": float(np.max(values)),
            }
        )
    return pd.DataFrame.from_records(records)


def _repeat_evidence(designs: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        designs,
        {"experiment_id", "design_id", "reduction_id", "is_reference", *_VALUE_COLUMNS},
        context="Reader designs",
    )
    work = designs.loc[~designs["is_reference"].astype(bool)].copy()
    counts = work.groupby(["reduction_id", "design_id"])["experiment_id"].nunique()
    repeated_keys = counts.loc[counts.gt(1)].index
    if repeated_keys.empty:
        raise ValueError("response-window comparison found no repeated designs.")
    repeated = work.set_index(["reduction_id", "design_id"]).loc[repeated_keys].reset_index()
    ranges = repeated.groupby(["reduction_id", "design_id"], sort=True)[list(_VALUE_COLUMNS)].agg(
        lambda values: float(np.ptp(values.to_numpy(dtype=float)))
    )
    maxima = ranges.max(axis=1).rename("maximum_channel_range").reset_index()
    return (
        maxima.groupby("reduction_id", sort=True)
        .agg(
            repeated_design_count=("design_id", "nunique"),
            repeat_median_maximum_channel_range=("maximum_channel_range", "median"),
            repeat_maximum_channel_range=("maximum_channel_range", "max"),
        )
        .reset_index()
    )


def _model_evidence(model_screen: pd.DataFrame, *, reduction_ids: tuple[str, ...]) -> pd.DataFrame:
    required = {"representation_id", "model_id", "weakest_required_ordering_spearman"}
    _require_columns(model_screen, required, context="model screen")
    rows = model_screen.loc[
        model_screen["representation_id"].astype(str).isin(reduction_ids)
        & model_screen["model_id"].astype(str).isin(_MODEL_IDS),
        list(required),
    ].copy()
    if rows.duplicated(subset=["representation_id", "model_id"]).any():
        raise ValueError("equal-footing model screen contains duplicate reduction/model rows.")
    pivot = rows.pivot(
        index="representation_id",
        columns="model_id",
        values="weakest_required_ordering_spearman",
    ).reindex(index=list(reduction_ids), columns=list(_MODEL_IDS))
    if pivot.isna().all(axis=None):
        raise ValueError("equal-footing model screen lacks the declared fixed diagnostic models.")
    pivot = pivot.rename_axis(index="reduction_id", columns=None).reset_index()
    return pivot.rename(columns={model_id: f"{model_id}_weakest_ordering_spearman" for model_id in _MODEL_IDS})


def _censoring_evidence(
    designs: pd.DataFrame,
    wells: pd.DataFrame,
    traces: pd.DataFrame,
    *,
    reduction_ids: tuple[str, ...],
) -> pd.DataFrame:
    design_fields = {
        f"{prefix}{state}_{suffix}"
        for prefix in ("r", "b")
        for state in ("00", "10", "01", "11")
        for suffix in ("has_policy_clipping", "has_instrument_overflow", "bound_kind")
    }
    event_fields = {
        f"{prefix}{state}_event_sensitivity_has_{cause}"
        for prefix in ("r", "b")
        for state in ("00", "10", "01", "11")
        for cause in ("policy_clipping", "instrument_overflow")
    }
    well_fields = {
        f"{signal}_{suffix}"
        for signal in ("response", "magnitude")
        for suffix in ("policy_clipped_point_count", "instrument_overflow_point_count", "bound_kind")
    }
    trace_fields = {"value_policy_clipped", "value_instrument_overflow", "value_bound_kind"}
    _require_columns(designs, {"reduction_id", *design_fields, *event_fields}, context="Reader v5 designs")
    _require_columns(wells, {"reduction_id", *well_fields}, context="Reader v5 wells")
    _require_columns(traces, trace_fields, context="Reader v5 traces")
    records: list[dict[str, object]] = []
    for reduction_id in reduction_ids:
        design_rows = designs.loc[designs["reduction_id"].astype(str).eq(reduction_id)]
        well_rows = wells.loc[wells["reduction_id"].astype(str).eq(reduction_id)]
        bound_columns = [f"{prefix}{state}_bound_kind" for prefix in ("r", "b") for state in ("00", "10", "01", "11")]
        policy_columns = [
            f"{prefix}{state}_has_policy_clipping" for prefix in ("r", "b") for state in ("00", "10", "01", "11")
        ]
        overflow_columns = [
            f"{prefix}{state}_has_instrument_overflow" for prefix in ("r", "b") for state in ("00", "10", "01", "11")
        ]
        event_policy_columns = [
            f"{prefix}{state}_event_sensitivity_has_policy_clipping"
            for prefix in ("r", "b")
            for state in ("00", "10", "01", "11")
        ]
        event_overflow_columns = [
            f"{prefix}{state}_event_sensitivity_has_instrument_overflow"
            for prefix in ("r", "b")
            for state in ("00", "10", "01", "11")
        ]
        event_censored = (
            design_rows.loc[:, event_policy_columns].astype(bool).to_numpy()
            | design_rows.loc[:, event_overflow_columns].astype(bool).to_numpy()
        )
        well_bound_columns = [f"{signal}_bound_kind" for signal in ("response", "magnitude")]
        records.append(
            {
                "reduction_id": reduction_id,
                "censoring_observability": "reader_v5_midpoint_and_event_bounds",
                "bounded_design_state_component_count": int(
                    design_rows.loc[:, bound_columns].astype(str).ne("exact").to_numpy().sum()
                ),
                "policy_clipped_design_state_component_count": int(
                    design_rows.loc[:, policy_columns].astype(bool).to_numpy().sum()
                ),
                "instrument_overflow_design_state_component_count": int(
                    design_rows.loc[:, overflow_columns].astype(bool).to_numpy().sum()
                ),
                "event_sensitivity_censored_design_state_component_count": int(event_censored.sum()),
                "event_sensitivity_policy_clipped_design_state_component_count": int(
                    design_rows.loc[:, event_policy_columns].astype(bool).to_numpy().sum()
                ),
                "event_sensitivity_instrument_overflow_design_state_component_count": int(
                    design_rows.loc[:, event_overflow_columns].astype(bool).to_numpy().sum()
                ),
                "bounded_well_signal_count": int(
                    well_rows.loc[:, well_bound_columns].astype(str).ne("exact").to_numpy().sum()
                ),
                "policy_clipped_well_point_count": int(
                    sum(well_rows[f"{signal}_policy_clipped_point_count"].sum() for signal in ("response", "magnitude"))
                ),
                "instrument_overflow_well_point_count": int(
                    sum(
                        well_rows[f"{signal}_instrument_overflow_point_count"].sum()
                        for signal in ("response", "magnitude")
                    )
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _require_columns(frame: pd.DataFrame, required: set[str], *, context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} lacks required columns: {missing}")


__all__ = ["build_response_window_evidence"]
