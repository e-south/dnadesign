"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/transforms_y/logic5_from_tidy_v1.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..exceptions import IngestError
from ..registries.ingest_transforms import register_ingest_transform


@register_ingest_transform("logic5_from_tidy_v1")
def transform_logic5_from_tidy_v1(
    df_tidy: pd.DataFrame,
    params: Dict[str, Any],
    setpoint_vector: List[float],
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: ['id', 'y'] where y is list[float] length 5:
    [v00, v10, v01, v11, effect_linear]. Strictly enforces a complete quartet.
    """
    schema = params["schema"]
    id_col = schema["design_id"]
    exp_col = schema["experiment_id"]
    time_col = schema["hours_post_induction"]
    rep_col = schema["replicate"]
    state_col = schema["logic_state"]
    chan_col = schema["reporter_channel"]
    val_col = schema["fluorescence_raw"]

    enforce_single_timepoint = params.get("enforce_single_timepoint", True)
    rep_agg = params.get("replicate_aggregation", "mean")
    rep_warn_threshold = params.get("replicate_warn_threshold", 3)

    pre = params.get("pre_processing", {})
    ratio_cfg = pre.get("compute_per_state_ratio", {})
    ratio_cfg.setdefault("input_numerator_channel", "yfp")
    ratio_cfg.setdefault("input_denominator_channel", "cfp")
    ratio_cfg.setdefault("division_epsilon", 1e-9)
    ratio_cfg.setdefault("apply_log2_to_ratio", True)

    logic_cfg = pre.get("build_logic_intensity_vector", {})
    logic_cfg.setdefault("expected_state_order", ["00", "10", "01", "11"])
    logic_cfg.setdefault("minmax_epsilon", 1e-9)
    logic_cfg.setdefault("equal_states_fallback", "uniform_quarters_and_warn")

    eff_cfg = pre.get("aggregate_effect_size_from_yfp", {})
    eff_cfg.setdefault("base_signal_channel", "yfp")

    # Normalize channel case
    df = df_tidy.copy()
    for col in (id_col, exp_col, time_col, rep_col, state_col, chan_col):
        if col not in df.columns:
            raise IngestError(f"Missing required column in tidy input: {col}")
    df[chan_col] = df[chan_col].astype(str).str.lower().str.strip()
    df[state_col] = df[state_col].astype(str).str.strip()

    # Enforce single timepoint per id if requested
    if enforce_single_timepoint:
        counts = df.groupby(id_col)[time_col].nunique()
        bad = counts[counts > 1]
        if not bad.empty:
            raise IngestError(
                f"Multiple timepoints for ids: {bad.index[:10].tolist()} ..."
            )

    # replicate aggregation: mean by [id, state, channel]
    df_grp = (
        df.groupby([id_col, state_col, chan_col], dropna=False)[val_col]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "value_mean", "count": "replicate_count"})
    )
    few_reps = df_grp[df_grp["replicate_count"] < rep_warn_threshold]
    if not few_reps.empty:
        print(
            f"[WARN] {len(few_reps)} (id,state,channel) groups have replicates < {rep_warn_threshold}"
        )

    yfp_name = str(ratio_cfg["input_numerator_channel"]).lower()
    cfp_name = str(ratio_cfg["input_denominator_channel"]).lower()
    eps = float(ratio_cfg["division_epsilon"])
    apply_log2 = bool(ratio_cfg["apply_log2_to_ratio"])

    piv = df_grp.pivot_table(
        index=[id_col, state_col], columns=chan_col, values="value_mean", aggfunc="mean"
    ).reset_index()

    required_states = logic_cfg["expected_state_order"]

    def compute_ratio(yfp: float, cfp: float) -> float:
        if cfp is None or np.isnan(cfp):
            return np.nan
        return yfp / max(cfp, eps)

    piv["ratio"] = piv.apply(
        lambda r: compute_ratio(r.get(yfp_name), r.get(cfp_name)), axis=1
    )
    if apply_log2:
        piv["ratio"] = np.log2(piv["ratio"])

    wide = piv.pivot_table(
        index=id_col, columns=state_col, values="ratio", aggfunc="mean"
    )
    for st in required_states:
        if st not in wide.columns:
            wide[st] = np.nan
    wide = wide[required_states].copy()

    if wide.isna().any(axis=None):
        bad_ids = wide[wide.isna().any(axis=1)].index.tolist()
        raise IngestError(
            f"Missing required (state,channel) measurements for {len(bad_ids)} design(s). Example: {bad_ids[:10]}"
        )

    mm_eps = float(logic_cfg["minmax_epsilon"])
    arr = wide.values.astype(float)
    lo = np.min(arr, axis=1, keepdims=True)
    hi = np.max(arr, axis=1, keepdims=True)
    span = np.maximum(hi - lo, mm_eps)
    logic_01 = (arr - lo) / span

    if logic_cfg.get("equal_states_fallback") == "uniform_quarters_and_warn":
        eq = (span <= mm_eps).ravel()
        if np.any(eq):
            print(
                f"[WARN] {np.sum(eq)} design(s) have nearly equal log2 ratios; using uniform quarters."
            )
            logic_01[eq, :] = 0.25

    # Effect: setpoint-weighted geometric mean of YFP over ON states (stored as linear units)
    s = np.array(list(setpoint_vector), dtype=float)
    if s.shape[0] != 4:
        raise IngestError("setpoint_vector must have length 4 for logic5.")
    s = np.clip(s, 0.0, None)
    w = np.zeros_like(s) if s.sum() == 0 else (s / s.sum())

    log2_yfp = (
        df[df[chan_col] == yfp_name]
        .groupby([id_col, state_col], dropna=False)[val_col]
        .apply(lambda x: np.log2(np.mean(x.values)))
        .unstack(fill_value=np.nan)
    )
    for st in required_states:
        if st not in log2_yfp.columns:
            log2_yfp[st] = np.nan
    log2_yfp = log2_yfp[required_states]
    if log2_yfp.isna().any(axis=None):
        bad = log2_yfp[log2_yfp.isna().any(axis=1)].index.tolist()
        raise IngestError(
            f"Missing YFP values for effect aggregation. Bad ids: {bad[:10]}"
        )

    m = (log2_yfp.values * w.reshape(1, -1)).sum(axis=1)
    effect_linear = np.power(2.0, m)

    out = pd.DataFrame(
        {
            "id": wide.index.astype(str),
            "y": [
                list(map(float, v))
                for v in np.concatenate(
                    [logic_01, effect_linear.reshape(-1, 1)], axis=1
                )
            ],
        }
    ).reset_index(drop=True)
    return out
