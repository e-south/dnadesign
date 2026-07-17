"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_allocation.py

Metric-neutral OPAL allocation preview for the two shadow objectives.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from dnadesign.opal import (
    SELECTION_ALLOCATION_PREVIEW_API_VERSION,
    preview_round_robin_next_best_unallocated,
)

from .multistate_behavior_protocol import MultistateBehaviorShadowProtocol

_EVIDENCE_ROLE = "same_fixed_prediction_sequence_deduplicated_allocation_preview_no_campaign_mutation"


def build_multistate_behavior_allocation_comparison(
    *,
    hard_behavior_detail: pd.DataFrame,
    candidate_records: pd.DataFrame,
    protocol: MultistateBehaviorShadowProtocol,
) -> pd.DataFrame:
    """Apply OPAL's production allocator to both fixed-prediction rankings."""

    detail = _validated_detail(hard_behavior_detail, protocol=protocol)
    candidates = _candidate_rows(candidate_records, ids=set(detail["id"].astype(str)))
    objective_specs = (
        (protocol.comparator_objective_name, protocol.comparator_score_channel, "hard_score", "hard_rank"),
        (protocol.objective_name, protocol.selector_output, "behavior_score", "behavior_rank"),
    )
    previews: dict[str, object] = {}
    allocated_by_objective: dict[str, pd.DataFrame] = {}
    for objective_name, score_channel, score_column, rank_column in objective_specs:
        view_rows = detail.loc[:, ["selection_view_id", "id", score_column, rank_column]].rename(
            columns={score_column: "score", rank_column: "rank"}
        )
        view_rows["top_k"] = protocol.prediction_raw_top_k
        preview = preview_round_robin_next_best_unallocated(
            candidate_rows=candidates.loc[:, ["id", "dedup_key"]],
            view_rows=view_rows,
            view_priority=protocol.completion_gate.allocation_view_priority,
        )
        if (
            int(preview.summary.get("final_unique_count", -1))
            != protocol.completion_gate.allocation_expected_unique_count
        ):
            raise ValueError(f"{objective_name} allocation did not produce the required unique candidate count.")
        allocated = preview.allocated.copy()
        allocated["objective_name"] = objective_name
        allocated["score_channel"] = score_channel
        allocated_by_objective[objective_name] = allocated
        previews[objective_name] = preview

    candidate_index = candidates.set_index("id")
    sequence_by_id = candidate_index["dedup_key"].to_dict()
    label_by_id = candidate_index["display_label"].to_dict()
    selected_sets = {
        objective_name: set(frame["id"].astype(str)) for objective_name, frame in allocated_by_objective.items()
    }
    records: list[dict[str, object]] = []
    source = detail.iloc[0]
    priority_json = json.dumps(protocol.completion_gate.allocation_view_priority, separators=(",", ":"))
    for objective_name, frame in allocated_by_objective.items():
        other_name = next(name for name in allocated_by_objective if name != objective_name)
        for row in frame.itertuples(index=False):
            candidate_id = str(row.id)
            sequence = str(sequence_by_id[candidate_id])
            records.append(
                {
                    "objective_name": objective_name,
                    "score_channel": str(row.score_channel),
                    "selection_view_id": str(row.selection_view_id),
                    "allocation_slot": int(row.allocation_slot),
                    "id": candidate_id,
                    "display_label": str(label_by_id[candidate_id]),
                    "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                    "rank": int(row.rank),
                    "score": float(row.score),
                    "raw_preference": int(row.rank) <= protocol.prediction_raw_top_k,
                    "selection_origin": str(row.selection_origin),
                    "also_allocated_by_other_objective": candidate_id in selected_sets[other_name],
                    "other_objective_name": other_name,
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
    result = (
        pd.DataFrame.from_records(records)
        .sort_values(
            ["objective_name", "selection_view_id", "allocation_slot"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    _verify_allocation_result(result, protocol=protocol)
    return result


def _validated_detail(
    frame: pd.DataFrame,
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> pd.DataFrame:
    required = {
        "id",
        "selection_view_id",
        "hard_score",
        "hard_rank",
        "behavior_score",
        "behavior_rank",
        "prediction_run_id",
        "prediction_source_sha256",
        "protocol_id",
        "protocol_source_sha256",
        "normalization_source_rows_sha256",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"allocation comparison lacks fixed-ranking fields: {missing}")
    rows = frame.loc[:, sorted(required)].copy()
    if rows.empty or rows.duplicated(subset=["id", "selection_view_id"]).any():
        raise ValueError("allocation comparison requires unique candidate/view rankings.")
    if set(rows["selection_view_id"].astype(str)) != set(protocol.completion_gate.allocation_view_priority):
        raise ValueError("allocation comparison selection views disagree with the completion gate.")
    expected_ids = set(rows.loc[rows["selection_view_id"].eq(protocol.target_views[0].id), "id"].astype(str))
    for view_id, view_rows in rows.groupby("selection_view_id", sort=False):
        if set(view_rows["id"].astype(str)) != expected_ids:
            raise ValueError(f"allocation comparison candidate coverage drifted for view {view_id!r}.")
        for rank_column in ("hard_rank", "behavior_rank"):
            ranks = pd.to_numeric(view_rows[rank_column], errors="raise").to_numpy(dtype=float)
            if not np.array_equal(np.sort(ranks.astype(int)), np.arange(1, len(view_rows) + 1)):
                raise ValueError(f"allocation comparison {rank_column} must be a complete ordinal ranking.")
    if set(rows["protocol_id"].astype(str)) != {protocol.protocol_id}:
        raise ValueError("allocation comparison protocol identity drifted.")
    return rows


def _candidate_rows(frame: pd.DataFrame, *, ids: set[str]) -> pd.DataFrame:
    required = {"id", "sequence"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"allocation comparison candidate records lack fields: {missing}")
    label_column = "usr_label__primary" if "usr_label__primary" in frame.columns else None
    columns = ["id", "sequence", *([label_column] if label_column else [])]
    if frame[["id", "sequence"]].isna().any().any():
        raise ValueError("allocation comparison candidate IDs and sequences must be non-null.")
    rows = frame.loc[frame["id"].astype(str).isin(ids), columns].copy()
    rows["id"] = rows["id"].astype(str)
    rows["sequence"] = rows["sequence"].astype(str)
    if set(rows["id"]) != ids or rows["id"].duplicated().any():
        raise ValueError("allocation comparison candidate records do not cover the fixed prediction pool exactly.")
    if rows["id"].str.strip().ne(rows["id"]).any() or rows["sequence"].str.strip().ne(rows["sequence"]).any():
        raise ValueError("allocation comparison candidate IDs and sequences must use exact non-padded values.")
    if rows["sequence"].eq("").any():
        raise ValueError("allocation comparison sequences must be nonempty.")
    if label_column is None:
        rows["display_label"] = rows["id"].str.slice(0, 10)
    else:
        labels = rows[label_column].astype("string")
        rows["display_label"] = labels.where(
            labels.notna() & labels.str.strip().ne(""),
            rows["id"].str.slice(0, 10),
        )
        rows = rows.drop(columns=label_column)
    return rows.rename(columns={"sequence": "dedup_key"}).reset_index(drop=True)


def _verify_allocation_result(frame: pd.DataFrame, *, protocol: MultistateBehaviorShadowProtocol) -> None:
    expected = protocol.completion_gate.allocation_expected_unique_count
    for objective_name, rows in frame.groupby("objective_name", sort=False):
        if len(rows) != expected or rows["id"].nunique() != expected or rows["sequence_sha256"].nunique() != expected:
            raise ValueError(f"allocation comparison objective {objective_name!r} is not sequence-unique.")
        if set(rows["selection_view_id"].astype(str)) != set(protocol.completion_gate.allocation_view_priority):
            raise ValueError(f"allocation comparison objective {objective_name!r} omits a selection view.")
        if not rows.groupby("selection_view_id").size().eq(protocol.prediction_raw_top_k).all():
            raise ValueError(f"allocation comparison objective {objective_name!r} does not fill every view quota.")


__all__ = ["build_multistate_behavior_allocation_comparison"]
