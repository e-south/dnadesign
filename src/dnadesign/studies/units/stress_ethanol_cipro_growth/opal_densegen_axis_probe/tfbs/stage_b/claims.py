"""Claim-readiness assessment for DenseGen TFBS Stage B learnability review."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

VALID_NEGATIVE_CONTROL = "VALID_AS_NEGATIVE_CONTROL"
PEER_STATUS_POSITIVE_EXCEEDS_NULL = "positive_exceeds_null"
PEER_STATUS_CONFOUND_CONTROL = "null_is_confound_control_only"
PEER_STATUS_NOT_SEPARATED = "not_separated_from_null"

CLAIM_READY = "READY_AS_VALID_NULL_LEARNABILITY_SIGNAL"
CLAIM_LIMITED_CONFOUND_CONTROL = "LIMITED_TO_CONFOUND_CONTROL_DIAGNOSTIC"
CLAIM_BLOCKED_NOT_SEPARATED = "BLOCKED_NOT_SEPARATED_FROM_NULL"
CLAIM_BLOCKED_NONPOSITIVE_TRAJECTORY = "BLOCKED_NONPOSITIVE_TRAJECTORY_DELTA"

_REQUIRED_COLUMNS = {
    "label_name",
    "label_family_id",
    "negative_control_claim_status",
    "peer_review_claim_status",
    "final_positive_minus_null_lift_ratio",
    "trapezoid_auc_positive_minus_null_lift_ratio",
}


def build_tfbs_stage_b_claim_assessment(pair_summary: pd.DataFrame) -> pd.DataFrame:
    """Return label-level claim boundaries for Stage B ML deliverable review."""

    missing = sorted(_REQUIRED_COLUMNS - set(pair_summary.columns))
    if missing:
        raise ValueError(f"Stage B claim assessment missing column(s): {missing}")
    if pair_summary.empty:
        raise ValueError("Stage B claim assessment requires at least one positive/null pair")

    rows: list[dict[str, Any]] = []
    for _, raw in pair_summary.iterrows():
        final_delta = _finite_float(raw["final_positive_minus_null_lift_ratio"], field="final lift delta")
        auc_delta = _finite_float(
            raw["trapezoid_auc_positive_minus_null_lift_ratio"],
            field="normalized trapezoid AUC lift delta",
        )
        status, boundary = _claim_status(
            negative_control_claim_status=str(raw["negative_control_claim_status"]),
            peer_review_claim_status=str(raw["peer_review_claim_status"]),
            final_delta=final_delta,
            auc_delta=auc_delta,
        )
        rows.append(
            {
                "label_name": str(raw["label_name"]),
                "label_family_id": str(raw["label_family_id"]),
                "negative_control_claim_status": str(raw["negative_control_claim_status"]),
                "peer_review_claim_status": str(raw["peer_review_claim_status"]),
                "final_positive_minus_null_lift_ratio": final_delta,
                "trapezoid_auc_positive_minus_null_lift_ratio": auc_delta,
                "claim_readiness_status": status,
                "claim_readiness_bool": bool(status == CLAIM_READY),
                "manuscript_claim_boundary": boundary,
            }
        )
    return pd.DataFrame(rows).sort_values(["claim_readiness_status", "label_name"]).reset_index(drop=True)


def summarize_tfbs_stage_b_claim_assessment(claims: pd.DataFrame) -> dict[str, Any]:
    """Return compact counts and label lists for the review summary JSON."""

    required = {"label_name", "claim_readiness_status", "claim_readiness_bool"}
    missing = sorted(required - set(claims.columns))
    if missing:
        raise ValueError(f"Stage B claim readiness summary missing column(s): {missing}")
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


def _claim_status(
    *,
    negative_control_claim_status: str,
    peer_review_claim_status: str,
    final_delta: float,
    auc_delta: float,
) -> tuple[str, str]:
    invalid_negative_control = negative_control_claim_status != VALID_NEGATIVE_CONTROL
    if peer_review_claim_status == PEER_STATUS_CONFOUND_CONTROL or invalid_negative_control:
        return (
            CLAIM_LIMITED_CONFOUND_CONTROL,
            "Report as a confound-control diagnostic only; do not claim valid-null learnability separation.",
        )
    if peer_review_claim_status == PEER_STATUS_NOT_SEPARATED:
        return (
            CLAIM_BLOCKED_NOT_SEPARATED,
            "Do not claim learnability separation; positive campaign did not exceed its matched valid null.",
        )
    if final_delta <= 0 or auc_delta <= 0:
        return (
            CLAIM_BLOCKED_NONPOSITIVE_TRAJECTORY,
            "Do not claim learnability separation; final and trajectory deltas must both be positive.",
        )
    if peer_review_claim_status == PEER_STATUS_POSITIVE_EXCEEDS_NULL:
        return (
            CLAIM_READY,
            "Eligible for a cautious synthetic-oracle learnability claim against a valid matched null.",
        )
    return (
        CLAIM_BLOCKED_NOT_SEPARATED,
        f"Do not claim learnability separation; unrecognized peer-review status {peer_review_claim_status!r}.",
    )


def _finite_float(value: Any, *, field: str) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="raise").iloc[0]
    as_float = float(out)
    if not math.isfinite(as_float):
        raise ValueError(f"Stage B claim assessment found non-finite {field}")
    return as_float
