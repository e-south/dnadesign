"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/model_evidence_manifest.py

Manifest projection for configured-model and challenger evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict

import numpy as np
import pandas as pd

from ..core.response_contracts import RESPONSE_REVIEW_SPEC
from ..evaluation.grouped_models import CAMPAIGN_MODEL_SCREEN_ID, DEFAULT_MODEL_SCREEN_SPECS


def build_model_evidence_manifest(
    model_screen: pd.DataFrame,
    *,
    primary_reduction_id: str,
    campaign_model_params: Mapping[str, object],
) -> dict[str, object]:
    """Keep the configured campaign model separate from fixed challengers."""

    promotion_rows = model_screen.loc[model_screen["promotion_eligible"].astype(bool)]
    campaign_rows = promotion_rows.loc[
        promotion_rows["model_role"].astype(str).eq("campaign_model")
        & promotion_rows["model_id"].astype(str).eq(CAMPAIGN_MODEL_SCREEN_ID)
        & promotion_rows["representation_id"].astype(str).eq(primary_reduction_id)
    ]
    if len(campaign_rows) != 1:
        raise ValueError(
            "response model screen must contain exactly one configured campaign-model row "
            f"for primary representation {primary_reduction_id!r}; found {len(campaign_rows)}."
        )
    campaign_model = campaign_rows.iloc[0]
    if str(campaign_model["target_transform"]) != "none":
        raise ValueError("configured campaign random forest must screen the primary target without transformation.")

    baseline_rows = promotion_rows.loc[
        promotion_rows["model_role"].astype(str).eq("baseline")
        & promotion_rows["representation_id"].astype(str).eq(primary_reduction_id)
    ]
    if len(baseline_rows) != 1:
        raise ValueError(
            "response model screen must contain exactly one baseline row "
            f"for primary representation {primary_reduction_id!r}; found {len(baseline_rows)}."
        )
    baseline = baseline_rows.iloc[0]

    challengers = promotion_rows.loc[
        promotion_rows["model_role"].astype(str).eq("fixed_challenger")
        & promotion_rows["all_target_view_metrics_finite"].astype(bool)
    ]
    if challengers.empty:
        raise ValueError("response model screen must contain at least one finite fixed challenger row.")
    best_challenger = challengers.sort_values(
        ["weakest_required_ordering_spearman", "median_channel_spearman"],
        ascending=False,
        kind="mergesort",
    ).iloc[0]
    return {
        "campaign_model_screen": _model_screen_record(
            campaign_model,
            posture="configured_campaign_model_not_promoted",
            configured_model_params=campaign_model_params,
        ),
        "best_fixed_model_screen": _model_screen_record(
            best_challenger,
            posture="descriptive_challenger_not_promoted",
        ),
        "baseline_model_screen": _model_screen_record(
            baseline,
            posture="reference_baseline_not_promoted",
        ),
        "prespecified_model_screens": [
            _model_screen_record(
                row,
                posture="retrospective_fixed_screen_not_promoted",
                configured_model_params=campaign_model_params if str(row["model_role"]) == "campaign_model" else None,
            )
            for _, row in promotion_rows.sort_values(
                ["representation_id", "model_role", "model_id"],
                kind="mergesort",
            ).iterrows()
        ],
        "fixed_model_definitions": [
            {"model_id": spec.id, **{key: value for key, value in asdict(spec).items() if key != "id"}}
            for spec in DEFAULT_MODEL_SCREEN_SPECS
        ],
        "model_support_basis": "configured_campaign_model",
        "model_support_ready": _model_support_ready(campaign_model),
        "challenger_policy": (
            "Fixed challengers are descriptive comparisons. They do not replace the configured campaign model "
            "without a separate prospective study decision."
        ),
    }


def _model_screen_record(
    row: pd.Series,
    *,
    posture: str,
    configured_model_params: Mapping[str, object] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "representation_id": str(row["representation_id"]),
        "model_id": str(row["model_id"]),
        "model_role": str(row["model_role"]),
        "target_transform": str(row["target_transform"]),
        "validation": str(row["validation"]),
        "weakest_target_view_response_separation_spearman": _finite_float_or_none(
            row["weakest_target_view_response_separation_spearman"]
        ),
        "weakest_target_view_feasibility_spearman": _finite_float_or_none(
            row["weakest_target_view_feasibility_spearman"]
        ),
        "weakest_required_ordering_spearman": _finite_float_or_none(row["weakest_required_ordering_spearman"]),
        "median_channel_spearman": _finite_float_or_none(row["median_channel_spearman"]),
        "minimum_channel_spearman": _finite_float_or_none(row["minimum_channel_spearman"]),
        "response_magnitude_mae": _finite_float_or_none(row["response_magnitude_mae"]),
        "minimum_defined_group_count": int(row["minimum_defined_group_count"]),
        "metric_scope": str(row["metric_scope"]),
        "posture": posture,
        "target_view_ordering": _target_view_ordering(row),
    }
    if configured_model_params is not None:
        record["configured_model_params"] = {
            str(key): value for key, value in sorted(configured_model_params.items(), key=lambda item: str(item[0]))
        }
    return record


def _target_view_ordering(row: pd.Series) -> dict[str, dict[str, float | int | None]]:
    response_suffix = "__response_separation_spearman"
    view_ids = sorted(
        str(column)[: -len(response_suffix)] for column in row.index if str(column).endswith(response_suffix)
    )
    if not view_ids:
        raise ValueError(f"model screen row {row['model_id']!r} has no target-view ordering fields.")
    return {
        view_id: {
            "response_separation_spearman": _finite_float_or_none(row[f"{view_id}{response_suffix}"]),
            "feasibility_spearman": _finite_float_or_none(row[f"{view_id}__feasibility_spearman"]),
            "defined_group_count": int(row[f"{view_id}__defined_group_count"]),
        }
        for view_id in view_ids
    }


def _finite_float_or_none(value: object) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _model_support_ready(row: pd.Series) -> bool:
    orderings = np.asarray(
        [
            row["weakest_target_view_response_separation_spearman"],
            row["weakest_target_view_feasibility_spearman"],
        ],
        dtype=float,
    )
    return bool(
        bool(row["all_target_view_metrics_finite"])
        and np.isfinite(orderings).all()
        and np.all(orderings >= RESPONSE_REVIEW_SPEC.model_min_within_group_spearman)
        and int(row["minimum_defined_group_count"]) >= RESPONSE_REVIEW_SPEC.model_min_defined_group_count
    )


__all__ = ["build_model_evidence_manifest"]
