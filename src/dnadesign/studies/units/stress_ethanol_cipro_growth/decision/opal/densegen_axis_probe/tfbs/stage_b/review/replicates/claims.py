"""Claim assessment from replicated Stage B endpoint summaries."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .contracts import (
    CLAIM_BLOCKED_INCOMPLETE_REPLICATES,
    CLAIM_BLOCKED_NONPOSITIVE_REPLICATED_ENDPOINT,
    CLAIM_BLOCKED_REPLICATE_NOT_READY,
    CLAIM_LIMITED_INVALID_NEGATIVE_CONTROL,
    CLAIM_READY_REPLICATED,
    TFBS_STAGE_B_DETERMINISTIC_REPLICATE_SEEDS,
)

_REQUIRED_COLUMNS = {
    "label_name",
    "label_family_id",
    "replicate_count",
    "replicate_seeds",
    "ready_replicate_count",
    "negative_control_ready_replicate_count",
    "final_positive_minus_null_lift_ratio_mean",
    "final_positive_minus_null_lift_ratio_median",
    "trapezoid_auc_positive_minus_null_lift_ratio_mean",
    "trapezoid_auc_positive_minus_null_lift_ratio_median",
}


def build_replicated_claim_assessment(endpoint_summary: pd.DataFrame) -> pd.DataFrame:
    """Return claim readiness from replicate-level endpoint summaries."""

    missing = sorted(_REQUIRED_COLUMNS - set(endpoint_summary.columns))
    if missing:
        raise ValueError(f"Stage B replicated claim assessment missing column(s): {missing}")
    if endpoint_summary.empty:
        raise ValueError("Stage B replicated claim assessment requires endpoint summary rows")
    rows: list[dict[str, Any]] = []
    for _, raw in endpoint_summary.iterrows():
        status, boundary = _claim_status(raw)
        rows.append(
            {
                "label_name": str(raw["label_name"]),
                "label_family_id": str(raw["label_family_id"]),
                "replicate_count": int(raw["replicate_count"]),
                "replicate_seeds": str(raw["replicate_seeds"]),
                "ready_replicate_count": int(raw["ready_replicate_count"]),
                "negative_control_ready_replicate_count": int(raw["negative_control_ready_replicate_count"]),
                "final_positive_minus_null_lift_ratio_mean": float(raw["final_positive_minus_null_lift_ratio_mean"]),
                "final_positive_minus_null_lift_ratio_median": float(
                    raw["final_positive_minus_null_lift_ratio_median"]
                ),
                "trapezoid_auc_positive_minus_null_lift_ratio_mean": float(
                    raw["trapezoid_auc_positive_minus_null_lift_ratio_mean"]
                ),
                "trapezoid_auc_positive_minus_null_lift_ratio_median": float(
                    raw["trapezoid_auc_positive_minus_null_lift_ratio_median"]
                ),
                "claim_readiness_status": status,
                "claim_readiness_bool": bool(status == CLAIM_READY_REPLICATED),
                "manuscript_claim_boundary": boundary,
            }
        )
    return pd.DataFrame(rows).sort_values(["claim_readiness_status", "label_name"]).reset_index(drop=True)


def summarize_replicated_claim_assessment(claims: pd.DataFrame) -> dict[str, Any]:
    """Return compact claim counts and label lists for the replicated review summary."""

    required = {"label_name", "claim_readiness_status", "claim_readiness_bool"}
    missing = sorted(required - set(claims.columns))
    if missing:
        raise ValueError(f"Stage B replicated claim readiness summary missing column(s): {missing}")
    ready = claims.loc[claims["claim_readiness_bool"].astype(bool)].copy()
    blocked = claims.loc[~claims["claim_readiness_bool"].astype(bool)].copy()
    status_counts = claims["claim_readiness_status"].value_counts().sort_index().to_dict()
    return {
        "ready_claim_count": int(len(ready)),
        "blocked_or_limited_claim_count": int(len(blocked)),
        "ready_labels": ready["label_name"].astype(str).tolist(),
        "blocked_or_limited_labels": blocked["label_name"].astype(str).tolist(),
        "claim_readiness_status_counts": {str(key): int(value) for key, value in status_counts.items()},
    }


def _claim_status(raw: pd.Series) -> tuple[str, str]:
    expected_count = len(TFBS_STAGE_B_DETERMINISTIC_REPLICATE_SEEDS)
    replicate_count = int(raw["replicate_count"])
    ready_count = int(raw["ready_replicate_count"])
    valid_negative_count = int(raw["negative_control_ready_replicate_count"])
    final_mean = float(raw["final_positive_minus_null_lift_ratio_mean"])
    final_median = float(raw["final_positive_minus_null_lift_ratio_median"])
    auc_mean = float(raw["trapezoid_auc_positive_minus_null_lift_ratio_mean"])
    auc_median = float(raw["trapezoid_auc_positive_minus_null_lift_ratio_median"])
    if replicate_count != expected_count:
        return (
            CLAIM_BLOCKED_INCOMPLETE_REPLICATES,
            f"Do not claim replicated separation; expected {expected_count} deterministic seed pairs.",
        )
    if valid_negative_count != replicate_count:
        return (
            CLAIM_LIMITED_INVALID_NEGATIVE_CONTROL,
            "Do not claim valid-null learnability separation; at least one replicate lacks a valid matched null.",
        )
    if ready_count != replicate_count:
        return (
            CLAIM_BLOCKED_REPLICATE_NOT_READY,
            "Do not claim replicated separation; at least one deterministic seed pair is not separated from null.",
        )
    if min(final_mean, final_median, auc_mean, auc_median) <= 0:
        return (
            CLAIM_BLOCKED_NONPOSITIVE_REPLICATED_ENDPOINT,
            "Do not claim replicated separation; mean and median final/AUC deltas must be positive.",
        )
    return (
        CLAIM_READY_REPLICATED,
        "Eligible for a cautious replicated construction-metadata learnability claim against matched controls.",
    )
