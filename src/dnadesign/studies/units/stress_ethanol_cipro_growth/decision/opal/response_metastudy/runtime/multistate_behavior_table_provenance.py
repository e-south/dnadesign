"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_table_provenance.py

Cross-table provenance and evidence-role checks for behavior bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .multistate_behavior_semantic_verification import BehaviorBundleSemantics

_SCORE_TABLES = (
    "observed_scores",
    "observed_coordinates",
    "bootstrap_scores",
    "event_sensitivity",
    "prediction_scores",
)
_PROTOCOL_TABLES = (*_SCORE_TABLES, "repeated_candidate_agreement", "hard_behavior_summary", "hard_behavior_detail")


def verify_behavior_table_provenance(
    tables: dict[str, pd.DataFrame],
    *,
    semantics: BehaviorBundleSemantics,
    objective_name: str,
    primary_reduction_id: str,
    comparator_semantics: str,
) -> None:
    """Require every evidence table to retain one coherent lineage."""

    for table_id in _PROTOCOL_TABLES:
        frame = tables[table_id]
        _require_single(frame, "protocol_id", semantics.protocol_id, table_id=table_id)
        _require_single(frame, "protocol_source_sha256", semantics.protocol_sha256, table_id=table_id)
        _require_single(
            frame,
            "normalization_source_rows_sha256",
            semantics.source_rows_sha256,
            table_id=table_id,
        )
    for table_id in _SCORE_TABLES:
        frame = tables[table_id]
        _require_single(frame, "objective_name", objective_name, table_id=table_id)
        _require_single(frame, "response_scale", semantics.response_scale, table_id=table_id, numeric=True)
        _require_single(
            frame,
            "signal_scale",
            semantics.signal_scale,
            table_id=table_id,
            numeric=True,
        )
        _require_single(frame, "status", "shadow_only", table_id=table_id)
        _require_single(frame, "campaign_activation", "prohibited", table_id=table_id)
        _require_single(frame, "synthesis_authorization", "prohibited", table_id=table_id)

    for table_id in ("prediction_scores", "hard_behavior_summary", "hard_behavior_detail"):
        frame = tables[table_id]
        _require_single(frame, "prediction_run_id", semantics.prediction_run_id, table_id=table_id)
        _require_single(
            frame,
            "prediction_source_sha256",
            semantics.prediction_source_sha256,
            table_id=table_id,
        )
    for table_id in ("hard_behavior_summary", "hard_behavior_detail"):
        _require_single(tables[table_id], "hard_score_semantics", comparator_semantics, table_id=table_id)

    for table_id in ("bootstrap_rank_draws", "bootstrap_rank_stability"):
        frame = tables[table_id]
        _require_single(
            frame,
            "ranking_method",
            "descending_score_then_ascending_candidate_experiment_unit_id",
            table_id=table_id,
        )
        _require_single(frame, "tie_semantics", "ordinal_rank_with_id_tiebreak", table_id=table_id)
        _require_single(
            frame,
            "evidence_role",
            "candidate_experiment_unit_rank_stability_no_label_aggregation_or_allocation",
            table_id=table_id,
        )
    for table_id in ("hard_behavior_summary", "hard_behavior_detail"):
        _require_single(
            tables[table_id],
            "ranking_method",
            "descending_score_then_ascending_candidate_id",
            table_id=table_id,
        )
        _require_single(tables[table_id], "tie_semantics", "ordinal_rank_with_id_tiebreak", table_id=table_id)
    _require_single(
        tables["hard_behavior_summary"],
        "evidence_role",
        "fixed_prediction_raw_candidate_ranking_comparison_no_sequence_allocation",
        table_id="hard_behavior_summary",
    )
    _require_single(
        tables["repeated_candidate_agreement"],
        "evidence_role",
        "repeat_agreement_only_no_label_aggregation_or_source_choice",
        table_id="repeated_candidate_agreement",
    )
    _verify_event_semantics(tables["event_sensitivity"])
    _verify_censor_semantics(tables["censor_exclusions"], primary_reduction_id=primary_reduction_id)
    _verify_required_numerics(tables)


def _verify_event_semantics(frame: pd.DataFrame) -> None:
    expected = {
        "ranking_method": "descending_score_then_ascending_candidate_experiment_unit_id",
        "tie_semantics": "ordinal_rank_with_id_tiebreak",
        "event_bound_semantics": "componentwise_conservative_not_joint_event_draw",
        "event_bound_probability_claim": "none",
        "event_censor_posture": "exact_unclipped_unoverflowed",
    }
    for field, value in expected.items():
        _require_single(frame, field, value, table_id="event_sensitivity")


def _verify_censor_semantics(frame: pd.DataFrame, *, primary_reduction_id: str) -> None:
    if frame.empty:
        return
    _require_single(frame, "reduction_id", primary_reduction_id, table_id="censor_exclusions")
    for column in (
        "component_is_nonexact",
        "has_policy_clipping",
        "has_instrument_overflow",
        "event_sensitivity_has_policy_clipping",
        "event_sensitivity_has_instrument_overflow",
    ):
        if frame[column].map(lambda value: not isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"censor exclusion {column} must contain exact booleans.")
    if not frame.groupby(["candidate_id", "reader_experiment_id"], sort=False)["component_is_nonexact"].any().all():
        raise ValueError("every excluded unit must contain at least one nonexact component.")
    allowed_bounds = {"exact", "lower", "upper", "indeterminate"}
    if unknown := sorted(set(frame["bound_kind"].astype(str)) - allowed_bounds):
        raise ValueError(f"censor exclusions contain unsupported bound kinds: {unknown}.")
    nonexact = frame["bound_kind"].astype(str).ne("exact")
    affected = frame["has_policy_clipping"].astype(bool) | frame["has_instrument_overflow"].astype(bool)
    if not frame["component_is_nonexact"].astype(bool).eq(nonexact).all() or not affected.eq(nonexact).all():
        raise ValueError("censor exclusion bound kind disagrees with exact/nonexact provenance.")


def _verify_required_numerics(tables: dict[str, pd.DataFrame]) -> None:
    columns = {
        "observed_scores": [
            "behavior_score",
            "hard_bottleneck_clearance",
            "response_family_score",
            "on_signal_family_score",
            "off_signal_suppression_family_score",
        ],
        "observed_coordinates": ["clearance", "bottleneck_weight"],
        "bootstrap_scores": [
            "behavior_score",
            "hard_bottleneck_clearance",
            "response_family_score",
            "on_signal_family_score",
            "off_signal_suppression_family_score",
        ],
        "event_sensitivity": [
            "behavior_score_central",
            "behavior_score_worst_envelope",
            "behavior_score_best_envelope",
            "behavior_score_envelope_width",
            "hard_bottleneck_worst_envelope",
            "hard_bottleneck_best_envelope",
        ],
        "repeated_candidate_agreement": [
            "behavior_score_min",
            "behavior_score_max",
            "behavior_score_range",
            "hard_bottleneck_min",
            "hard_bottleneck_max",
            "hard_bottleneck_range",
        ],
        "prediction_scores": [
            "behavior_score",
            "hard_bottleneck_clearance",
            "response_family_score",
            "on_signal_family_score",
            "off_signal_suppression_family_score",
        ],
        "hard_behavior_detail": ["hard_score", "behavior_score", "hard_rank", "behavior_rank"],
    }
    for table_id, names in columns.items():
        if not np.isfinite(tables[table_id].loc[:, names].to_numpy(dtype=float)).all():
            raise ValueError(f"table {table_id!r} contains non-finite required numeric evidence.")


def _require_single(
    frame: pd.DataFrame,
    column: str,
    expected: object,
    *,
    table_id: str,
    numeric: bool = False,
) -> None:
    if column not in frame:
        raise ValueError(f"table {table_id!r} lacks provenance column {column!r}.")
    if frame.empty:
        return
    if numeric:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.allclose(values, float(expected), rtol=1e-12, atol=0.0):
            raise ValueError(f"table {table_id!r} provenance {column!r} drifted.")
    elif set(frame[column].astype(str)) != {str(expected)}:
        raise ValueError(f"table {table_id!r} provenance {column!r} drifted.")


__all__ = ["verify_behavior_table_provenance"]
