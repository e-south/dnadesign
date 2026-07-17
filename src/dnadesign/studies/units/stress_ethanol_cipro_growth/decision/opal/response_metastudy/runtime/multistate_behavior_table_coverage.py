"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_table_coverage.py

Exact state, view, unit, and draw coverage checks for behavior tables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .multistate_behavior_semantic_verification import BehaviorBundleSemantics


def verify_behavior_table_coverage(
    tables: dict[str, pd.DataFrame],
    *,
    semantics: BehaviorBundleSemantics,
    scale_quantile: float,
    quantile_method: str,
) -> None:
    """Require every table to cover the exact semantic product it claims."""

    response = tables["normalization_response_resolution"]
    signal = tables["normalization_signal_resolution"]
    observed = tables["observed_scores"]
    unit_ids = set(observed["id"].astype(str))
    if len(unit_ids) != semantics.unit_count:
        raise ValueError("observed score unit identities disagree with the cohort receipt.")
    _require_identity_map(response, observed, context="response normalization")
    _require_identity_map(signal, observed, context="signal normalization")
    if set(response["id"].astype(str)) != unit_ids or set(signal["id"].astype(str)) != unit_ids:
        raise ValueError("normalization table units disagree with observed score units.")
    _verify_normalization_coverage(
        response,
        signal,
        semantics=semantics,
        scale_quantile=scale_quantile,
        quantile_method=quantile_method,
    )

    _require_view_product(observed, ids=unit_ids, semantics=semantics, context="observed scores")
    _require_view_product(
        tables["event_sensitivity"],
        ids=unit_ids,
        semantics=semantics,
        context="event sensitivity",
    )
    _require_bootstrap_product(tables["bootstrap_scores"], ids=unit_ids, semantics=semantics)
    _require_coordinate_product(tables["observed_coordinates"], ids=unit_ids, semantics=semantics)
    for table_id in ("observed_coordinates", "bootstrap_scores", "event_sensitivity"):
        _require_identity_map(tables[table_id], observed, context=table_id.replace("_", " "))

    predictions = tables["prediction_scores"]
    prediction_ids = set(predictions["id"].astype(str))
    if len(prediction_ids) != semantics.prediction_count:
        raise ValueError("prediction score identities disagree with the prediction receipt.")
    _require_view_product(predictions, ids=prediction_ids, semantics=semantics, context="prediction scores")
    _require_view_product(
        tables["hard_behavior_detail"],
        ids=prediction_ids,
        semantics=semantics,
        context="hard behavior detail",
    )
    _require_exact_views(tables["hard_behavior_summary"], semantics=semantics, context="hard behavior summary")
    _require_exact_views(
        tables["bootstrap_rank_stability"],
        semantics=semantics,
        context="bootstrap rank stability",
    )
    _require_rank_draw_product(tables["bootstrap_rank_draws"], semantics=semantics)
    _verify_censor_product(
        tables["censor_exclusions"],
        observed=observed,
        semantics=semantics,
    )


def _verify_normalization_coverage(
    response: pd.DataFrame,
    signal: pd.DataFrame,
    *,
    semantics: BehaviorBundleSemantics,
    scale_quantile: float,
    quantile_method: str,
) -> None:
    expected_pairs = set(semantics.response_pairs)
    expected_states = set(semantics.state_ids)
    declared_by_pair = _declared_views_by_pair(semantics)
    for unit_id, rows in response.groupby("id", sort=False):
        pairs = set(zip(rows["state_a"].astype(str), rows["state_b"].astype(str), strict=True))
        if pairs != expected_pairs or rows.duplicated(subset=["state_a", "state_b"]).any():
            raise ValueError(f"response normalization pair coverage drifted for unit {unit_id!r}.")
        for row in rows.itertuples(index=False):
            if str(row.declared_by_selection_views) != ",".join(declared_by_pair[(str(row.state_a), str(row.state_b))]):
                raise ValueError("response normalization declaring-view provenance drifted.")
    for unit_id, rows in signal.groupby("id", sort=False):
        states = set(rows["state_id"].astype(str))
        if states != expected_states or rows["state_id"].astype(str).duplicated().any():
            raise ValueError(f"signal normalization state coverage drifted for unit {unit_id!r}.")
    for table_id, frame in (
        ("response", response),
        ("signal", signal),
    ):
        values = frame["bootstrap_sd"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"{table_id} normalization SD evidence must be finite and nonnegative.")
        if not frame["bootstrap_samples"].eq(semantics.bootstrap_samples).all():
            raise ValueError(f"{table_id} normalization bootstrap support drifted.")
    observed_response = float(
        np.quantile(response["bootstrap_sd"].to_numpy(dtype=float), scale_quantile, method=quantile_method)
    )
    observed_signal = float(
        np.quantile(signal["bootstrap_sd"].to_numpy(dtype=float), scale_quantile, method=quantile_method)
    )
    if not np.isclose(observed_response, semantics.response_scale, rtol=1e-12, atol=0.0):
        raise ValueError("response scale does not derive from persisted normalization rows.")
    if not np.isclose(observed_signal, semantics.signal_scale, rtol=1e-12, atol=0.0):
        raise ValueError("signal scale does not derive from persisted pDual-10-relative fluorescence rows.")


def _require_view_product(
    frame: pd.DataFrame,
    *,
    ids: set[str],
    semantics: BehaviorBundleSemantics,
    context: str,
) -> None:
    if frame.duplicated(subset=["id", "selection_view_id"]).any() or set(frame["id"].astype(str)) != ids:
        raise ValueError(f"{context} contains duplicate or missing unit identities.")
    expected_views = set(semantics.view_ids)
    if any(set(rows["selection_view_id"].astype(str)) != expected_views for _, rows in frame.groupby("id")):
        raise ValueError(f"{context} does not cover every declared selection view per id.")


def _require_bootstrap_product(
    frame: pd.DataFrame,
    *,
    ids: set[str],
    semantics: BehaviorBundleSemantics,
) -> None:
    if frame.duplicated(subset=["id", "selection_view_id", "draw_index"]).any():
        raise ValueError("bootstrap scores contain duplicate semantic keys.")
    if set(frame["id"].astype(str)) != ids:
        raise ValueError("bootstrap score units disagree with observed scores.")
    expected = {
        (view_id, draw_index) for view_id in semantics.view_ids for draw_index in range(semantics.bootstrap_samples)
    }
    for unit_id, rows in frame.groupby("id", sort=False):
        observed = set(zip(rows["selection_view_id"].astype(str), rows["draw_index"].astype(int), strict=True))
        if observed != expected:
            raise ValueError(f"bootstrap score view/draw coverage drifted for unit {unit_id!r}.")


def _require_coordinate_product(
    frame: pd.DataFrame,
    *,
    ids: set[str],
    semantics: BehaviorBundleSemantics,
) -> None:
    if frame.duplicated(subset=["id", "selection_view_id", "coordinate_label"]).any():
        raise ValueError("observed coordinates contain duplicate semantic keys.")
    if set(frame["id"].astype(str)) != ids:
        raise ValueError("observed coordinate units disagree with observed scores.")
    expected = {
        view_id: _coordinate_labels(semantics.state_ids, mask) for view_id, mask in semantics.view_masks.items()
    }
    for (unit_id, view_id), rows in frame.groupby(["id", "selection_view_id"], sort=False):
        if set(rows["coordinate_label"].astype(str)) != expected.get(str(view_id), set()):
            raise ValueError(f"coordinate coverage drifted for unit/view {unit_id!r}/{view_id!r}.")


def _require_exact_views(frame: pd.DataFrame, *, semantics: BehaviorBundleSemantics, context: str) -> None:
    if frame["selection_view_id"].astype(str).duplicated().any() or set(frame["selection_view_id"].astype(str)) != set(
        semantics.view_ids
    ):
        raise ValueError(f"{context} must contain exactly one row per declared selection view.")


def _require_rank_draw_product(frame: pd.DataFrame, *, semantics: BehaviorBundleSemantics) -> None:
    if frame.duplicated(subset=["selection_view_id", "draw_index"]).any():
        raise ValueError("bootstrap rank draws contain duplicate semantic keys.")
    expected_draws = set(range(semantics.bootstrap_samples))
    for view_id, rows in frame.groupby("selection_view_id", sort=False):
        if str(view_id) not in semantics.view_ids or set(rows["draw_index"].astype(int)) != expected_draws:
            raise ValueError("bootstrap rank draw coverage drifted by selection view.")
    if set(frame["selection_view_id"].astype(str)) != set(semantics.view_ids):
        raise ValueError("bootstrap rank draws omit a declared selection view.")


def _verify_censor_product(
    frame: pd.DataFrame,
    *,
    observed: pd.DataFrame,
    semantics: BehaviorBundleSemantics,
) -> None:
    components = {f"{prefix}{state_id}" for prefix in ("r", "b") for state_id in semantics.state_ids}
    units = frame[["candidate_id", "reader_experiment_id"]].astype(str).drop_duplicates()
    if len(units) != semantics.excluded_nonexact_unit_count:
        raise ValueError("censor exclusions do not cover the declared excluded-unit count.")
    exact_units = set(observed[["candidate_id", "reader_experiment_id"]].astype(str).itertuples(index=False, name=None))
    excluded_units = set(units.itertuples(index=False, name=None))
    if exact_units & excluded_units:
        raise ValueError("censor exclusions overlap exact observed units.")
    for key, rows in frame.groupby(["candidate_id", "reader_experiment_id"], sort=False):
        if set(rows["component"].astype(str)) != components or rows["component"].astype(str).duplicated().any():
            raise ValueError(f"censor component coverage drifted for excluded unit {key!r}.")
        if rows["design_id"].astype(str).nunique() != 1:
            raise ValueError(f"censor design identity drifted for excluded unit {key!r}.")


def _require_identity_map(source: pd.DataFrame, observed: pd.DataFrame, *, context: str) -> None:
    source_map = source.loc[:, ["id", "candidate_id", "reader_experiment_id"]].astype(str).drop_duplicates()
    observed_map = observed.loc[:, ["id", "candidate_id", "reader_experiment_id"]].astype(str).drop_duplicates()
    if (
        source_map.sort_values("id")
        .reset_index(drop=True)
        .equals(observed_map.sort_values("id").reset_index(drop=True))
        is False
    ):
        raise ValueError(f"{context} candidate-experiment identity disagrees with observed scores.")


def _declared_views_by_pair(semantics: BehaviorBundleSemantics) -> dict[tuple[str, str], tuple[str, ...]]:
    index = {state_id: position for position, state_id in enumerate(semantics.state_ids)}
    result: dict[tuple[str, str], list[str]] = {pair: [] for pair in semantics.response_pairs}
    for view_id, mask in semantics.view_masks.items():
        on = [semantics.state_ids[position] for position, value in enumerate(mask) if value == 1]
        off = [semantics.state_ids[position] for position, value in enumerate(mask) if value == 0]
        for left in on:
            for right in off:
                pair = tuple(sorted((left, right), key=index.__getitem__))
                result[pair].append(view_id)  # type: ignore[index]
    return {pair: tuple(sorted(view_ids)) for pair, view_ids in result.items()}


def _coordinate_labels(state_ids: tuple[str, ...], mask: tuple[int, ...]) -> set[str]:
    on = [state_ids[index] for index, value in enumerate(mask) if value == 1]
    off = [state_ids[index] for index, value in enumerate(mask) if value == 0]
    return {
        *(f"response:{left}>{right}" for left in on for right in off),
        *(f"on_signal:{state}" for state in on),
        *(f"off_signal_suppression:{state}" for state in off),
    }


__all__ = ["verify_behavior_table_coverage"]
