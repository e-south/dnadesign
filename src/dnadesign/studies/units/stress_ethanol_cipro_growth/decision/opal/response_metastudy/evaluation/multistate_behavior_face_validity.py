"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_face_validity.py

Observed control projection for biological face-validity review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from .multistate_behavior_protocol import MultistateBehaviorShadowProtocol


def build_behavior_face_validity(
    observed_scores: pd.DataFrame,
    measurements: pd.DataFrame,
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> pd.DataFrame:
    """Project existing SpyP/sulAp controls without creating a score threshold."""

    measurement_fields = {"candidate_id", "design_id", "reader_experiment_id", "reduction_id"}
    if missing := sorted(measurement_fields - set(measurements.columns)):
        raise ValueError(f"behavior face-validity measurements lack fields: {missing}")
    score_fields = {
        "id",
        "candidate_id",
        "reader_experiment_id",
        "selection_view_id",
        "behavior_score",
        "hard_bottleneck_clearance",
        "response_family_score",
        "on_signal_family_score",
        "off_signal_suppression_family_score",
        "limiting_coordinate",
        "all_reference_directions_met",
    }
    if missing := sorted(score_fields - set(observed_scores.columns)):
        raise ValueError(f"behavior face-validity scores lack fields: {missing}")
    primary = measurements.loc[
        measurements["reduction_id"].astype(str).eq(protocol.primary_reduction_id),
        list(measurement_fields),
    ].copy()
    scores = observed_scores.copy()
    scores["observed_unit_rank"] = scores.groupby("selection_view_id", group_keys=False).apply(
        _unit_ranks,
        include_groups=False,
    )
    records: list[dict[str, object]] = []
    for control in protocol.completion_gate.face_validity_controls:
        identities = primary.loc[primary["design_id"].astype(str).eq(control.design_id)]
        if identities.empty:
            raise ValueError(
                f"behavior face-validity control {control.design_id!r} is absent from primary Reader evidence."
            )
        merged = identities.merge(
            scores.loc[scores["selection_view_id"].astype(str).eq(control.selection_view_id)],
            on=["candidate_id", "reader_experiment_id"],
            how="left",
            validate="one_to_one",
        )
        if merged["behavior_score"].isna().any():
            raise ValueError(f"behavior face-validity control {control.design_id!r} lacks an observed score.")
        for row in merged.itertuples(index=False):
            records.append(
                {
                    "selection_view_id": control.selection_view_id,
                    "design_id": control.design_id,
                    "display_label": control.display_label,
                    "id": str(row.id),
                    "candidate_id": str(row.candidate_id),
                    "reader_experiment_id": str(row.reader_experiment_id),
                    "observed_unit_rank": int(row.observed_unit_rank),
                    "observed_unit_count": int(
                        scores["selection_view_id"].astype(str).eq(control.selection_view_id).sum()
                    ),
                    "behavior_score": float(row.behavior_score),
                    "hard_bottleneck_clearance": float(row.hard_bottleneck_clearance),
                    "response_family_score": float(row.response_family_score),
                    "on_signal_family_score": float(row.on_signal_family_score),
                    "off_signal_suppression_family_score": float(row.off_signal_suppression_family_score),
                    "limiting_coordinate": str(row.limiting_coordinate),
                    "all_reference_directions_met": bool(row.all_reference_directions_met),
                    "evidence_role": protocol.completion_gate.face_validity_evidence_role,
                }
            )
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["selection_view_id", "reader_experiment_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _unit_ranks(frame: pd.DataFrame) -> pd.Series:
    ordered = frame.sort_values(["behavior_score", "id"], ascending=[False, True], kind="mergesort")
    result = pd.Series(index=frame.index, dtype=int)
    result.loc[ordered.index] = range(1, len(ordered) + 1)
    return result.astype(int)


__all__ = ["build_behavior_face_validity"]
