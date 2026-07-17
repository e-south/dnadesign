"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_allocation_verification.py

Fail-closed replay of metric-neutral allocation-preview evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json

import pandas as pd

from dnadesign.opal import (
    SELECTION_ALLOCATION_PREVIEW_API_VERSION,
    preview_round_robin_next_best_unallocated,
)

from ..evaluation.multistate_behavior_protocol import MultistateBehaviorShadowProtocol

_EVIDENCE_ROLE = "same_fixed_prediction_sequence_deduplicated_allocation_preview_no_campaign_mutation"


def verify_allocation_comparison(
    tables: dict[str, pd.DataFrame],
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> None:
    """Replay both allocations and every user-facing identity/provenance field."""

    observed = tables["allocation_comparison"]
    detail = tables["hard_behavior_detail"]
    vectors = tables["prediction_vectors"]
    expected_objectives = {protocol.comparator_objective_name, protocol.objective_name}
    if set(observed["objective_name"].astype(str)) != expected_objectives:
        raise ValueError("allocation comparison objectives are incomplete or unexpected.")
    vector_index = vectors.set_index("id")
    candidate_rows = vectors.loc[:, ["id", "sequence_sha256"]].rename(columns={"sequence_sha256": "dedup_key"})
    specs = (
        (
            protocol.comparator_objective_name,
            protocol.comparator_score_channel,
            "hard_score",
            "hard_rank",
        ),
        (protocol.objective_name, protocol.selector_output, "behavior_score", "behavior_rank"),
    )
    allocations: dict[str, pd.DataFrame] = {}
    for objective, score_channel, score, rank in specs:
        view_rows = detail.loc[:, ["selection_view_id", "id", score, rank]].rename(
            columns={score: "score", rank: "rank"}
        )
        view_rows["top_k"] = protocol.prediction_raw_top_k
        replay = preview_round_robin_next_best_unallocated(
            candidate_rows=candidate_rows,
            view_rows=view_rows,
            view_priority=protocol.completion_gate.allocation_view_priority,
        ).allocated
        replay["objective_name"] = objective
        replay["score_channel"] = score_channel
        allocations[objective] = replay

    expected_records: list[dict[str, object]] = []
    priority_json = json.dumps(protocol.completion_gate.allocation_view_priority, separators=(",", ":"))
    source = detail.iloc[0]
    selected = {name: set(rows["id"].astype(str)) for name, rows in allocations.items()}
    for objective, rows in allocations.items():
        other = next(name for name in allocations if name != objective)
        for row in rows.itertuples(index=False):
            candidate_id = str(row.id)
            candidate = vector_index.loc[candidate_id]
            expected_records.append(
                {
                    "objective_name": objective,
                    "score_channel": str(row.score_channel),
                    "selection_view_id": str(row.selection_view_id),
                    "allocation_slot": int(row.allocation_slot),
                    "id": candidate_id,
                    "display_label": str(candidate["display_label"]),
                    "sequence_sha256": str(candidate["sequence_sha256"]),
                    "rank": int(row.rank),
                    "score": float(row.score),
                    "raw_preference": int(row.rank) <= protocol.prediction_raw_top_k,
                    "selection_origin": str(row.selection_origin),
                    "also_allocated_by_other_objective": candidate_id in selected[other],
                    "other_objective_name": other,
                    "allocation_api_version": SELECTION_ALLOCATION_PREVIEW_API_VERSION,
                    "allocation_strategy": protocol.completion_gate.allocation_strategy,
                    "allocation_deduplicate_by": protocol.completion_gate.allocation_deduplicate_by,
                    "allocation_view_priority_json": priority_json,
                    "expected_unique_count": protocol.completion_gate.allocation_expected_unique_count,
                    "prediction_run_id": str(source.prediction_run_id),
                    "prediction_source_sha256": str(source.prediction_source_sha256),
                    "protocol_id": protocol.protocol_id,
                    "protocol_source_sha256": f"sha256:{protocol.source_sha256}",
                    "normalization_source_rows_sha256": str(source.normalization_source_rows_sha256),
                    "evidence_role": _EVIDENCE_ROLE,
                }
            )
    expected = pd.DataFrame.from_records(expected_records).loc[:, list(observed.columns)]
    keys = ["objective_name", "selection_view_id", "allocation_slot"]
    try:
        pd.testing.assert_frame_equal(
            observed.sort_values(keys).reset_index(drop=True),
            expected.sort_values(keys).reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError(
            "allocation comparison does not replay exactly from fixed rankings and identity rows."
        ) from exc


__all__ = ["verify_allocation_comparison"]
