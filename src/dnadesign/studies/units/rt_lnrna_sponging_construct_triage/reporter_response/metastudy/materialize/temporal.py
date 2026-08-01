"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/materialize/temporal.py

Temporal selection, trace reduction, and growth-phase summaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ... import EndpointReduction, ReporterResponseContractError
from ...profile import Reduction
from ...temporal import TemporalSelectedRow, reduce_temporal_input_trace
from ..condition_ontology import ReporterResponseConditionOntology
from ..contracts.profile import GrowthPhaseStratum
from ..contracts.protocol import MetastudyProtocol


def _growth_phase_strata(
    frame: pd.DataFrame,
    *,
    reduction: Reduction,
    ontology: ReporterResponseConditionOntology,
    protocol: MetastudyProtocol,
) -> tuple[GrowthPhaseStratum, ...] | str:
    """Derive study-owned growth-phase position from observed normalizer traces."""

    if isinstance(reduction, EndpointReduction):
        return ()
    normalizer = frame.loc[frame["channel"].eq(ontology.normalizer_channel)].copy()
    if normalizer.empty:
        return "phase_not_estimable_normalizer_missing"
    normalizer["time"] = pd.to_numeric(normalizer["time"], errors="coerce")
    normalizer["value"] = pd.to_numeric(normalizer["value"], errors="coerce")
    if normalizer["time"].isna().any() or not np.isfinite(normalizer["time"].to_numpy(dtype=float)).all():
        return "phase_not_estimable_nonfinite_od"
    definitions = ontology.by_treatment_label
    rows: list[GrowthPhaseStratum] = []
    for treatment, treatment_rows in normalizer.groupby("treatment", sort=True, dropna=False):
        label = str(treatment)
        definition = definitions.get(label)
        if definition is None:
            return "phase_not_estimable_condition_not_declared"
        trace = (
            treatment_rows.groupby("time", as_index=False, sort=True)["value"]
            .median()
            .sort_values("time", kind="stable")
        )
        times = trace["time"].to_numpy(dtype=float)
        values = trace["value"].to_numpy(dtype=float)
        # The scale spans the trace, so rejected quality observations must not
        # influence it. Keep the raw trace for candidate-boundary checks below.
        scale_rows = treatment_rows.loc[
            ~treatment_rows["value_policy_clipped"].astype(bool)
            & ~treatment_rows["value_instrument_overflow"].astype(bool)
            & treatment_rows["value_bound_kind"].astype(str).eq("exact")
            & np.isfinite(treatment_rows["value"].to_numpy(dtype=float))
            & treatment_rows["value"].gt(0.0)
        ]
        scale_trace = (
            scale_rows.groupby("time", as_index=False, sort=True)["value"].median().sort_values("time", kind="stable")
        )
        if scale_trace.empty:
            return "phase_not_estimable_positive_slope_scale"
        scale_times = scale_trace["time"].to_numpy(dtype=float)
        scale_values = scale_trace["value"].to_numpy(dtype=float)
        first_start = math.ceil(float(scale_times.min()) - 1e-9)
        last_start = math.floor(float(scale_times.max()) - protocol.growth_phase_slope_window_h + 1e-9)
        slopes = tuple(
            value
            for start in range(first_start, last_start + 1)
            if (
                value := _log_normalizer_slope(
                    scale_times,
                    scale_values,
                    start_h=float(start),
                    protocol=protocol,
                )
            )
            is not None
            and value > 0.0
        )
        if not slopes:
            return "phase_not_estimable_positive_slope_scale"
        scale = float(
            np.quantile(
                np.asarray(slopes, dtype=float),
                protocol.growth_phase_scale_quantile,
                method="linear",
            )
        )
        start_slope = _log_normalizer_slope(
            times,
            values,
            start_h=reduction.recorded_start_time_h,
            protocol=protocol,
        )
        end_slope = _log_normalizer_slope(
            times,
            values,
            start_h=reduction.recorded_end_time_h - protocol.growth_phase_slope_window_h,
            protocol=protocol,
        )
        if start_slope is None or end_slope is None or scale <= 0.0:
            return "phase_not_estimable_temporal_support"
        rows.append(
            GrowthPhaseStratum(
                condition_id=definition.condition_id,
                normalized_start_slope=start_slope / scale,
                normalized_end_slope=end_slope / scale,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.condition_id))


def _log_normalizer_slope(
    times: np.ndarray,
    values: np.ndarray,
    *,
    start_h: float,
    protocol: MetastudyProtocol,
) -> float | None:
    end_h = start_h + protocol.growth_phase_slope_window_h
    mask = (times >= start_h - 1e-9) & (times <= end_h + 1e-9)
    if int(mask.sum()) < protocol.growth_phase_minimum_slope_points:
        return None
    selected_times = times[mask]
    if len(set(selected_times.tolist())) != len(selected_times):
        return None
    selected_values = values[mask]
    if not np.isfinite(selected_values).all() or np.any(selected_values <= 0.0):
        return None
    slope = float(np.polyfit(selected_times, np.log(selected_values), 1)[0])
    return slope if math.isfinite(slope) else None


def _select_reduction(frame: pd.DataFrame, reduction: Reduction) -> pd.DataFrame:
    time = pd.to_numeric(frame["time"], errors="coerce")
    if time.isna().any():
        return frame.iloc[0:0]
    if isinstance(reduction, EndpointReduction):
        return frame.loc[(time - reduction.recorded_time_h).abs().le(1e-9)].copy()
    return frame.loc[
        time.ge(reduction.recorded_start_time_h - 1e-9) & time.le(reduction.recorded_end_time_h + 1e-9)
    ].copy()


def _condition_summary(
    frame: pd.DataFrame,
    ontology: ReporterResponseConditionOntology,
    *,
    reduction: Reduction,
    protocol: MetastudyProtocol,
):
    channels = (ontology.reporter_channel, ontology.normalizer_channel, ontology.ratio_channel)
    if frame.empty or set(frame["channel"].astype(str)) != set(channels):
        return None
    by_observation: dict[str, tuple[float, float, float]] = {}
    temporal_policy = reduction.temporal_policy
    if temporal_policy is None:
        return None
    for observation_identity in sorted(set(frame["position"].astype(str))):
        observation = frame.loc[frame["position"].astype(str).eq(observation_identity)]
        time_sets = {
            channel: tuple(sorted(pd.to_numeric(observation.loc[observation["channel"].eq(channel), "time"]).tolist()))
            for channel in channels
        }
        if not time_sets[channels[0]] or len(set(time_sets.values())) != 1:
            return None
        observed_times = time_sets[channels[0]]
        if not all(math.isfinite(value) for value in observed_times):
            return None
        if len(observed_times) != len(set(observed_times)):
            return None
        values: list[float] = []
        for channel in channels:
            channel_rows = observation.loc[observation["channel"].eq(channel)]
            trace = tuple(
                TemporalSelectedRow(
                    observation_identity=observation_identity,
                    time_h=float(row.time),
                    value=float(row.value),
                    value_policy_clipped=bool(getattr(row, "value_policy_clipped", False)),
                    value_instrument_overflow=bool(getattr(row, "value_instrument_overflow", False)),
                    value_bound_kind=str(getattr(row, "value_bound_kind", "exact")),  # type: ignore[arg-type]
                )
                for row in channel_rows.itertuples(index=False)
            )
            try:
                values.append(
                    reduce_temporal_input_trace(
                        trace,
                        policy=temporal_policy,
                        within_acquisition_statistic="median",
                    )
                )
            except ReporterResponseContractError:
                return None
        if not all(math.isfinite(value) for value in values) or values[1] <= 0.0:
            return None
        by_observation[observation_identity] = (values[0], values[1], values[2])
    if len(by_observation) < protocol.minimum_within_acquisition_observations_per_stratum:
        return None
    reporter = _reduce((row[0] for row in by_observation.values()), "median")
    normalizer = _reduce((row[1] for row in by_observation.values()), "median")
    ratio_values = tuple(row[2] for row in by_observation.values())
    ratio = (
        reporter / normalizer
        if isinstance(reduction, EndpointReduction)
        else _reduce(ratio_values, protocol.time_summary_statistic)
    )
    return reporter, normalizer, ratio, len(by_observation), max(ratio_values) - min(ratio_values)


def _contains_censored_values(frame: pd.DataFrame) -> bool:
    return bool(
        frame["value_policy_clipped"].astype(bool).any()
        or frame["value_instrument_overflow"].astype(bool).any()
        or frame["value_bound_kind"].astype(str).ne("exact").any()
    )


def _reduce(values: Iterable[float], statistic: str) -> float:
    rows = tuple(float(value) for value in values)
    return float(statistics.median(rows) if statistic == "median" else statistics.mean(rows))
