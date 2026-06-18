"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/review/replicates/aggregation.py

Replicate-level aggregation for Stage B realized-label review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import pandas as pd

from ..frames import pair_summary_frame, trajectory_frame
from ..io import campaign_rows, pair_rows
from .claims import build_replicated_claim_assessment
from .contracts import TfbsStageBReplicateManifest

ENDPOINT_METRICS = (
    "positive_final_lift_ratio",
    "null_final_lift_ratio",
    "final_positive_minus_null_lift_ratio",
    "positive_mean_round_lift_ratio",
    "null_mean_round_lift_ratio",
    "mean_round_positive_minus_null_lift_ratio",
    "positive_trapezoid_auc_lift_ratio",
    "null_trapezoid_auc_lift_ratio",
    "trapezoid_auc_positive_minus_null_lift_ratio",
)


def build_replicated_review_frames(
    entries: Sequence[TfbsStageBReplicateManifest],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return trajectory, pair summary, endpoint summary, and replicated claims."""

    trajectory_frames: list[pd.DataFrame] = []
    pair_summary_frames: list[pd.DataFrame] = []
    for entry in entries:
        campaigns = campaign_rows(entry.manifest)
        trajectory = trajectory_frame(campaigns, rounds=int(entry.manifest["rounds"]))
        trajectory["source_config_manifest_path"] = str(entry.path)
        trajectory["replicate_seed"] = int(entry.seed)
        pair_summary = pair_summary_frame(
            trajectory,
            campaigns=campaigns,
            pairs=pair_rows(entry.manifest),
        )
        pair_summary["source_config_manifest_path"] = str(entry.path)
        pair_summary["replicate_seed"] = int(entry.seed)
        trajectory_frames.append(trajectory)
        pair_summary_frames.append(pair_summary)
    trajectory_all = pd.concat(trajectory_frames, ignore_index=True).sort_values(
        ["label_name", "oracle_role", "seed", "round"]
    )
    pair_summary_all = pd.concat(pair_summary_frames, ignore_index=True).sort_values(["label_name", "seed"])
    _reject_duplicate_replicate_rows(trajectory_all, pair_summary_all)
    endpoint_summary = endpoint_summary_frame(pair_summary_all)
    claim_assessment = build_replicated_claim_assessment(endpoint_summary)
    return trajectory_all, pair_summary_all, endpoint_summary, claim_assessment


def endpoint_summary_frame(pair_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate endpoint metrics from one row per label/replicate seed pair."""

    required = {"label_name", "label_family_id", "split_id", "seed", *ENDPOINT_METRICS}
    missing = sorted(required - set(pair_summary.columns))
    if missing:
        raise ValueError(f"Stage B replicated endpoint summary missing column(s): {missing}")
    rows: list[dict[str, Any]] = []
    grouped = pair_summary.groupby(["label_name", "label_family_id", "split_id"], sort=True, dropna=False)
    for (label_name, family_id, split_id), group in grouped:
        seeds = tuple(sorted(int(seed) for seed in group["seed"].tolist()))
        row: dict[str, Any] = {
            "label_name": str(label_name),
            "label_family_id": str(family_id),
            "split_id": str(split_id),
            "replicate_count": int(len(group)),
            "replicate_seeds": ",".join(str(seed) for seed in seeds),
            "ready_replicate_count": int(_ready_replicates(group).sum()),
            "negative_control_ready_replicate_count": int(
                (group["negative_control_claim_status"].astype(str) == "VALID_AS_NEGATIVE_CONTROL").sum()
            ),
            "positive_exceeds_null_replicate_count": int(
                (group["peer_review_claim_status"].astype(str) == "positive_exceeds_null").sum()
            ),
        }
        for metric in ENDPOINT_METRICS:
            row.update(_metric_stats(group[metric], prefix=metric))
        rows.append(row)
    if not rows:
        raise ValueError("Stage B replicated endpoint summary requires at least one label group")
    return pd.DataFrame(rows).sort_values("label_name").reset_index(drop=True)


def _reject_duplicate_replicate_rows(trajectory: pd.DataFrame, pair_summary: pd.DataFrame) -> None:
    trajectory_key = ["label_name", "oracle_role", "seed", "round"]
    if trajectory.duplicated(trajectory_key).any():
        sample = (
            trajectory.loc[trajectory.duplicated(trajectory_key, keep=False), trajectory_key].head(5).to_dict("records")
        )
        raise ValueError(f"Stage B replicated trajectory contains duplicate seed/round rows: {sample}")
    pair_key = ["label_name", "split_id", "seed"]
    if pair_summary.duplicated(pair_key).any():
        sample = pair_summary.loc[pair_summary.duplicated(pair_key, keep=False), pair_key].head(5).to_dict("records")
        raise ValueError(f"Stage B replicated pair summary contains duplicate label/seed rows: {sample}")


def _ready_replicates(group: pd.DataFrame) -> pd.Series:
    return (
        (group["negative_control_claim_status"].astype(str) == "VALID_AS_NEGATIVE_CONTROL")
        & (group["peer_review_claim_status"].astype(str) == "positive_exceeds_null")
        & (pd.to_numeric(group["final_positive_minus_null_lift_ratio"], errors="raise") > 0)
        & (pd.to_numeric(group["trapezoid_auc_positive_minus_null_lift_ratio"], errors="raise") > 0)
    )


def _metric_stats(values: pd.Series, *, prefix: str) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.empty:
        raise ValueError(f"Stage B replicated endpoint metric {prefix!r} has no values")
    stats = {
        f"{prefix}_mean": float(numeric.mean()),
        f"{prefix}_median": float(numeric.median()),
        f"{prefix}_q25": float(numeric.quantile(0.25)),
        f"{prefix}_q75": float(numeric.quantile(0.75)),
        f"{prefix}_min": float(numeric.min()),
        f"{prefix}_max": float(numeric.max()),
    }
    if not all(math.isfinite(value) for value in stats.values()):
        raise ValueError(f"Stage B replicated endpoint metric {prefix!r} contains non-finite summary values")
    return stats
