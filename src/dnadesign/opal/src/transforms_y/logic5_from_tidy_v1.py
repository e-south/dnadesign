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
from ..registries.transforms_y import register_ingest_transform


@register_ingest_transform("logic5_from_tidy_v1")
def transform_logic5_from_tidy_v1(
    df_tidy: pd.DataFrame,
    params: Dict[str, Any],
    setpoint_vector: List[float],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ingest tidy fluorescence reads into vec8 labels per design id.

    Returns: (df_id_y, meta)
      - df_id_y with columns ['id','y'] where y is list[float] length 8:
        [v00, v10, v01, v11, y00_log2, y10_log2, y01_log2, y11_log2]
      - meta: dict with counts, warnings, and parameters actually used.
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

    # intensity: we store per-state log2(YFP) as y* (reference anchor optional)
    int_cfg = pre.get("intensity_log2_per_state", {})
    int_cfg.setdefault("signal_channel", "yfp")
    int_cfg.setdefault("log2_epsilon", 1e-9)

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

    # Per-state log2 intensity from YFP channel (reference-normalization optional; not applied here)
    log2_yfp = (
        df[df[chan_col] == yfp_name]
        .groupby([id_col, state_col], dropna=False)[val_col]
        .apply(lambda x: np.log2(max(np.mean(x.values), float(int_cfg["log2_epsilon"])) ))
        .unstack(fill_value=np.nan)
    )
    for st in required_states:
        if st not in log2_yfp.columns:
            log2_yfp[st] = np.nan
    log2_yfp = log2_yfp[required_states]
    if log2_yfp.isna().any(axis=None):
        bad = log2_yfp[log2_yfp.isna().any(axis=1)].index.tolist()
        raise IngestError(
            f"Missing YFP values for intensity computation. Bad ids: {bad[:10]}"
        )

    # Assemble vec8: [v00,v10,v01,v11,y00*,y10*,y01*,y11*]
    ystar = log2_yfp[required_states].values.astype(float)
    y_vec8 = np.concatenate([logic_01, ystar], axis=1)

    out = pd.DataFrame(
        {
            "id": wide.index.astype(str),
            "y": [list(map(float, row)) for row in y_vec8],
        }
    ).reset_index(drop=True)

    meta = {
        "num_ids": int(len(out)),
        "replicate_warn_threshold": int(rep_warn_threshold),
        "ratio": {
            "numerator": yfp_name,
            "denominator": cfp_name,
            "division_epsilon": float(eps),
            "apply_log2": bool(apply_log2),
        },
        "logic": {
            "expected_state_order": required_states,
            "minmax_epsilon": float(mm_eps),
        },
        "intensity": {
            "signal_channel": yfp_name,
            "log2_epsilon": float(int_cfg["log2_epsilon"]),
        },
        "warnings": [],
    }
    return out, meta
