"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_normalization_protocol.py

Normalization sub-contract for the behavior shadow protocol.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .multistate_behavior_protocol_fields import (
    BehaviorProtocolError,
    nonempty_string,
    require_exact_fields,
    require_literal,
)


@dataclass(frozen=True)
class BehaviorNormalizationProtocol:
    cohort_id: str
    unit: str
    response_scale_basis: str
    signal_scale_basis: str
    pair_deduplication: Literal["unique_unordered_state_pair_union"]
    scale_quantile: float
    quantile_method: Literal["linear"]
    minimum_bootstrap_draws: int
    normalized_temperature: float
    bootstrap_role: str
    event_time_role: str
    repeat_role: str
    censor_role: str


def parse_behavior_normalization_protocol(
    payload: dict[str, object],
    evidence: dict[str, object],
    *,
    normalized_temperature: float,
) -> BehaviorNormalizationProtocol:
    """Parse the fixed, study-owned assay-resolution convention."""

    require_exact_fields(
        payload,
        {
            "cohort_id",
            "unit",
            "response_scale_basis",
            "signal_scale_basis",
            "pair_deduplication",
            "scale_quantile",
            "quantile_method",
            "minimum_bootstrap_draws",
        },
        context="normalization",
    )
    require_exact_fields(
        evidence,
        {"bootstrap", "event_time", "repeat", "censor"},
        context="evidence_roles",
    )
    literals = (
        (payload, "pair_deduplication", "unique_unordered_state_pair_union", "normalization"),
        (payload, "quantile_method", "linear", "normalization"),
        (payload, "cohort_id", "exact_primary_reader_candidate_experiments_v1", "normalization"),
        (payload, "unit", "reader_candidate_experiment", "normalization"),
        (
            payload,
            "response_scale_basis",
            "reader_joint_bootstrap_sd_of_declared_on_off_response_pairs",
            "normalization",
        ),
        (
            payload,
            "signal_scale_basis",
            "reader_joint_bootstrap_sd_of_each_reference_relative_state",
            "normalization",
        ),
        (
            evidence,
            "bootstrap",
            "normalization_and_candidate_experiment_unit_rank_sensitivity_no_top_k",
            "evidence_roles",
        ),
        (evidence, "event_time", "separate_sensitivity_evidence", "evidence_roles"),
        (evidence, "repeat", "separate_disagreement_evidence", "evidence_roles"),
        (evidence, "censor", "exact_only_normalization_cohort", "evidence_roles"),
    )
    for record, field, expected, context in literals:
        require_literal(record, field, expected, context=context)
    scale_quantile = float(payload["scale_quantile"])
    if not math.isfinite(scale_quantile) or not 0.5 <= scale_quantile < 1.0:
        raise BehaviorProtocolError("normalization.scale_quantile must be in [0.5, 1).")
    minimum_draws = payload["minimum_bootstrap_draws"]
    if isinstance(minimum_draws, bool) or not isinstance(minimum_draws, int) or minimum_draws < 100:
        raise BehaviorProtocolError("normalization.minimum_bootstrap_draws must be an integer >= 100.")
    return BehaviorNormalizationProtocol(
        cohort_id=nonempty_string(payload["cohort_id"], field="cohort_id"),
        unit=nonempty_string(payload["unit"], field="unit"),
        response_scale_basis=nonempty_string(
            payload["response_scale_basis"],
            field="response_scale_basis",
        ),
        signal_scale_basis=nonempty_string(
            payload["signal_scale_basis"],
            field="signal_scale_basis",
        ),
        pair_deduplication="unique_unordered_state_pair_union",
        scale_quantile=scale_quantile,
        quantile_method="linear",
        minimum_bootstrap_draws=minimum_draws,
        normalized_temperature=normalized_temperature,
        bootstrap_role=nonempty_string(evidence["bootstrap"], field="bootstrap"),
        event_time_role="separate_sensitivity_evidence",
        repeat_role="separate_disagreement_evidence",
        censor_role="exact_only_normalization_cohort",
    )


__all__ = ["BehaviorNormalizationProtocol", "parse_behavior_normalization_protocol"]
