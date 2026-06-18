"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/review/plots/statistics.py

Replicate summaries for Stage B realized-label review plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd


def replicate_round_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return mean and sample-SD lift by acquisition round."""

    summary = (
        frame.groupby("round", as_index=False)["lift"]
        .agg(
            lift_mean="mean",
            lift_sample_sd=lambda values: values.std(ddof=1),
            replicate_count="count",
        )
        .sort_values("round")
    )
    summary["lift_sample_sd"] = summary["lift_sample_sd"].fillna(0.0)
    summary["lift_lower"] = summary["lift_mean"] - summary["lift_sample_sd"]
    summary["lift_upper"] = summary["lift_mean"] + summary["lift_sample_sd"]
    return summary


def trajectory_replicate_count(frame: pd.DataFrame) -> int:
    """Return the maximum replicate count per role/round cell."""

    return int(frame.groupby(["oracle_role", "round"], dropna=False).size().max())


def role_sort_key(role: object) -> tuple[int, str]:
    """Sort positive before matched null and unknown roles last."""

    role_text = str(role)
    if role_text == "positive":
        return (0, role_text)
    if role_text == "matched_null":
        return (1, role_text)
    return (2, role_text)


def replicate_column(frame: pd.DataFrame) -> str:
    """Return the best available replicate identity column for a trajectory."""

    if "seed" in frame.columns:
        return "seed"
    if "campaign_key" in frame.columns:
        return "campaign_key"
    return "round"


def seed_lift_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    """Summarize initial-batch lift across replicate identities."""

    if "seed" in frame.columns:
        replicate_column_name = "seed"
    elif "campaign_key" in frame.columns:
        replicate_column_name = "campaign_key"
    else:
        replicate_column_name = None
    if replicate_column_name is None:
        values = pd.to_numeric(frame["seed_true_lift_ratio"], errors="raise")
    else:
        values = pd.to_numeric(
            frame.drop_duplicates([replicate_column_name])["seed_true_lift_ratio"],
            errors="raise",
        )
    return replicate_value_summary(values)


def replicate_value_summary(values: pd.Series) -> dict[str, float | int]:
    """Return mean, sample SD, and replicate count for one value vector."""

    numeric = pd.to_numeric(values, errors="raise")
    if numeric.empty:
        raise ValueError("Stage B replicate summary requires at least one value")
    return {
        "mean": float(numeric.mean()),
        "sample_sd": float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
        "replicate_count": int(len(numeric)),
    }
