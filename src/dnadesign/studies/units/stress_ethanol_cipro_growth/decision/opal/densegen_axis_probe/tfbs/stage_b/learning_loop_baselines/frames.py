"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/learning_loop_baselines/frames.py

Tidy frame builders for frozen learning-loop baseline reviews.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..review.frames import normalized_trapezoid_auc


def cumulative_lift_trajectory(
    selections: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    label_name: str,
    pool_baseline: float,
    campaign_key: str,
    oracle_role: str,
    scientific_control_role: str | None = None,
    seed: int,
    selection_k: int,
) -> pd.DataFrame:
    """Return cumulative acquired-budget lift rows for one campaign."""

    if pool_baseline <= 0:
        raise ValueError("cumulative lift requires positive pool_baseline")
    missing_selections = sorted({"round", "id", "selection_source"} - set(selections.columns))
    if missing_selections:
        raise ValueError(f"cumulative lift selections missing column(s): {missing_selections}")
    missing_labels = sorted({"id", label_name} - set(labels.columns))
    if missing_labels:
        raise ValueError(f"cumulative lift labels missing column(s): {missing_labels}")

    label_frame = labels.loc[:, ["id", label_name]].copy()
    label_frame["id"] = label_frame["id"].astype(str)
    if label_frame["id"].duplicated().any():
        duplicates = label_frame.loc[label_frame["id"].duplicated(), "id"].drop_duplicates().head(5).tolist()
        raise ValueError(f"cumulative lift labels contain duplicate id(s): {duplicates}")
    label_by_id = dict(zip(label_frame["id"], pd.to_numeric(label_frame[label_name], errors="raise"), strict=True))

    selected = selections.loc[:, ["round", "id", "selection_source"]].copy()
    selected["id"] = selected["id"].astype(str)
    selected["round"] = pd.to_numeric(selected["round"], errors="raise").astype(int)
    if selected["id"].duplicated().any():
        duplicates = selected.loc[selected["id"].duplicated(), "id"].drop_duplicates().head(5).tolist()
        raise ValueError(f"cumulative lift selections contain duplicate id(s): {duplicates}")
    missing_ids = sorted(set(selected["id"]) - set(label_by_id))
    if missing_ids:
        raise ValueError(f"cumulative lift selected id(s) missing from labels: {missing_ids[:5]}")

    rows: list[dict[str, Any]] = []
    seen: list[str] = []
    for round_index, sub in selected.sort_values(["round"]).groupby("round", sort=True):
        round_ids = sub["id"].tolist()
        if len(round_ids) != int(selection_k):
            raise ValueError(f"round {round_index} selected {len(round_ids)} rows; expected {int(selection_k)}")
        seen.extend(round_ids)
        values = np.array([float(label_by_id[candidate_id]) for candidate_id in seen], dtype=float)
        mean = float(values.mean())
        rows.append(
            {
                "campaign_key": str(campaign_key),
                "label_name": str(label_name),
                "oracle_role": str(oracle_role),
                "scientific_control_role": str(scientific_control_role or ""),
                "seed": int(seed),
                "round": int(round_index),
                "selection_source": str(sub["selection_source"].iloc[0]),
                "selected_count": int(len(round_ids)),
                "selection_k": int(selection_k),
                "cumulative_selected_count": int(len(seen)),
                "cumulative_label_sum": float(values.sum()),
                "cumulative_label_mean": mean,
                "pool_baseline": float(pool_baseline),
                "cumulative_lift_ratio": mean / float(pool_baseline),
            }
        )
    return pd.DataFrame(rows)


def endpoint_summary_frame(
    trajectory: pd.DataFrame,
    *,
    pairs: list[dict[str, Any]],
) -> pd.DataFrame:
    """Return campaign-source endpoints and paired positive/control deltas."""

    campaign_endpoints = _campaign_endpoint_rows(trajectory)
    pair_rows = []
    for pair in pairs:
        label_name = str(pair["label_name"])
        seed = int(pair["seed"])
        positive_key = str(pair["positive_campaign_key"])
        control_key = str(pair["null_campaign_key"])
        for source in sorted(campaign_endpoints["selection_source"].unique().tolist()):
            pos = _single_endpoint(campaign_endpoints, campaign_key=positive_key, selection_source=source)
            control = _single_endpoint(campaign_endpoints, campaign_key=control_key, selection_source=source)
            pair_rows.append(
                {
                    "row_type": "pair_endpoint",
                    "label_name": label_name,
                    "seed": seed,
                    "selection_source": source,
                    "positive_campaign_key": positive_key,
                    "control_campaign_key": control_key,
                    "positive_final_cumulative_lift_ratio": float(pos["final_cumulative_lift_ratio"]),
                    "control_final_cumulative_lift_ratio": float(control["final_cumulative_lift_ratio"]),
                    "positive_minus_control_final_cumulative_lift_ratio": float(
                        pos["final_cumulative_lift_ratio"] - control["final_cumulative_lift_ratio"]
                    ),
                    "positive_cumulative_auc_lift_ratio": float(pos["cumulative_auc_lift_ratio"]),
                    "control_cumulative_auc_lift_ratio": float(control["cumulative_auc_lift_ratio"]),
                    "positive_minus_control_cumulative_auc_lift_ratio": float(
                        pos["cumulative_auc_lift_ratio"] - control["cumulative_auc_lift_ratio"]
                    ),
                    "positive_final_cumulative_label_mean": float(pos["final_cumulative_label_mean"]),
                    "control_final_cumulative_label_mean": float(control["final_cumulative_label_mean"]),
                    "pool_baseline": float(pos["pool_baseline"]),
                }
            )
    return pd.concat([campaign_endpoints, pd.DataFrame(pair_rows)], ignore_index=True, sort=False)


def claim_interpretation_frame(endpoint_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replicate-level active/frozen/known-label-reference evidence by TFBS label."""

    pairs = endpoint_summary.loc[endpoint_summary["row_type"] == "pair_endpoint"].copy()
    if pairs.empty:
        raise ValueError("Frozen replay interpretation requires pair_endpoint rows")
    rows: list[dict[str, Any]] = []
    for label_name, sub_label in pairs.groupby("label_name", sort=True):
        active = _source_pairs(sub_label, "active_retraining")
        frozen = _source_pairs(sub_label, "frozen_round0")
        known_label = _source_pairs(sub_label, "known_label_ranking")
        merged = active.merge(
            frozen,
            on=["label_name", "seed"],
            suffixes=("_active", "_frozen"),
            validate="one_to_one",
        )
        merged = merged.merge(
            known_label,
            on=["label_name", "seed"],
            validate="one_to_one",
        )
        active_gain = (
            merged["positive_final_cumulative_lift_ratio_active"]
            - merged["positive_final_cumulative_lift_ratio_frozen"]
        )
        active_auc_gain = (
            merged["positive_cumulative_auc_lift_ratio_active"] - merged["positive_cumulative_auc_lift_ratio_frozen"]
        )
        active_minus_control = merged["positive_minus_control_final_cumulative_lift_ratio_active"]
        frozen_minus_control = merged["positive_minus_control_final_cumulative_lift_ratio_frozen"]
        known_label_final = merged["positive_final_cumulative_lift_ratio"]
        known_label_auc = merged["positive_cumulative_auc_lift_ratio"]
        active_fraction_of_known_label = _safe_ratio(
            merged["positive_final_cumulative_lift_ratio_active"],
            known_label_final,
        )
        active_gain_recovered = _safe_ratio(
            merged["positive_final_cumulative_lift_ratio_active"] - 1.0,
            known_label_final - 1.0,
        )
        rows.append(
            {
                "label_name": str(label_name),
                "replicate_count": int(len(merged)),
                "active_final_cumulative_lift_mean": float(
                    merged["positive_final_cumulative_lift_ratio_active"].mean()
                ),
                "frozen_final_cumulative_lift_mean": float(
                    merged["positive_final_cumulative_lift_ratio_frozen"].mean()
                ),
                "active_minus_frozen_final_cumulative_lift_mean": float(active_gain.mean()),
                "active_minus_frozen_final_cumulative_lift_sample_sd": _sample_sd(active_gain),
                "active_minus_frozen_cumulative_auc_lift_mean": float(active_auc_gain.mean()),
                "active_minus_frozen_cumulative_auc_lift_sample_sd": _sample_sd(active_auc_gain),
                "known_label_final_cumulative_lift_mean": float(known_label_final.mean()),
                "known_label_cumulative_auc_lift_mean": float(known_label_auc.mean()),
                "active_fraction_of_known_label_final_lift_mean": float(active_fraction_of_known_label.mean()),
                "active_fraction_of_known_label_final_lift_sample_sd": _sample_sd(active_fraction_of_known_label),
                "active_fraction_of_known_label_gain_recovered_mean": float(active_gain_recovered.mean()),
                "active_fraction_of_known_label_gain_recovered_sample_sd": _sample_sd(active_gain_recovered),
                "seeds_supporting_adaptive_final_gain": int((active_gain > 0).sum()),
                "seeds_supporting_adaptive_auc_gain": int((active_auc_gain > 0).sum()),
                "active_positive_minus_control_final_lift_mean": float(active_minus_control.mean()),
                "frozen_positive_minus_control_final_lift_mean": float(frozen_minus_control.mean()),
                "interpretation_status": _interpretation_status(active_gain, active_auc_gain),
            }
        )
    return pd.DataFrame(rows)


def _campaign_endpoint_rows(trajectory: pd.DataFrame) -> pd.DataFrame:
    required = {
        "campaign_key",
        "label_name",
        "oracle_role",
        "seed",
        "selection_source",
        "round",
        "cumulative_lift_ratio",
        "cumulative_label_mean",
        "pool_baseline",
    }
    missing = sorted(required - set(trajectory.columns))
    if missing:
        raise ValueError(f"Frozen replay endpoint summary missing column(s): {missing}")
    rows: list[dict[str, Any]] = []
    group_cols = ["campaign_key", "selection_source"]
    for (campaign_key, selection_source), sub in trajectory.groupby(group_cols, sort=True):
        ordered = sub.sort_values("round")
        final = ordered.iloc[-1]
        auc_frame = ordered.rename(columns={"cumulative_lift_ratio": "selected_true_lift_ratio"})
        rows.append(
            {
                "row_type": "campaign_endpoint",
                "campaign_key": str(campaign_key),
                "label_name": str(final["label_name"]),
                "oracle_role": str(final["oracle_role"]),
                "seed": int(final["seed"]),
                "selection_source": str(selection_source),
                "final_round": int(final["round"]),
                "final_cumulative_selected_count": int(final["cumulative_selected_count"]),
                "final_cumulative_label_mean": float(final["cumulative_label_mean"]),
                "final_cumulative_lift_ratio": float(final["cumulative_lift_ratio"]),
                "cumulative_auc_lift_ratio": normalized_trapezoid_auc(auc_frame),
                "pool_baseline": float(final["pool_baseline"]),
            }
        )
    return pd.DataFrame(rows)


def _single_endpoint(frame: pd.DataFrame, *, campaign_key: str, selection_source: str) -> pd.Series:
    sub = frame.loc[
        (frame["campaign_key"].astype(str) == str(campaign_key))
        & (frame["selection_source"].astype(str) == str(selection_source))
    ]
    if len(sub) != 1:
        raise ValueError(
            f"Expected one endpoint for campaign={campaign_key!r} source={selection_source!r}; found {len(sub)}"
        )
    return sub.iloc[0]


def _source_pairs(frame: pd.DataFrame, selection_source: str) -> pd.DataFrame:
    out = frame.loc[frame["selection_source"].astype(str) == str(selection_source)].copy()
    if out.empty:
        raise ValueError(f"Frozen replay interpretation missing source rows for {selection_source}")
    return out


def _interpretation_status(active_gain: pd.Series, active_auc_gain: pd.Series) -> str:
    if int((active_gain > 0).sum()) >= 2 and float(active_gain.mean()) > 0:
        return "ADAPTIVE_GAIN_SUPPORTED"
    if int((active_auc_gain > 0).sum()) >= 2 and float(active_auc_gain.mean()) > 0:
        return "ADAPTIVE_AUC_GAIN_ONLY"
    return "REPRESENTATION_SIGNAL_WITHOUT_CLEAR_ADAPTIVE_GAIN"


def _sample_sd(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="raise")
    return float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numer = pd.to_numeric(numerator, errors="raise").astype(float)
    denom = pd.to_numeric(denominator, errors="raise").astype(float)
    return numer.divide(denom).where(denom != 0.0, 0.0)
