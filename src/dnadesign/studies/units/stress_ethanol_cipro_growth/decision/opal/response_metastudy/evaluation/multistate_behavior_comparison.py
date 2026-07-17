"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_comparison.py

Hard-versus-smooth and repeated-experiment behavior comparisons.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class HardBehaviorComparison:
    summary: pd.DataFrame
    detail: pd.DataFrame


def compare_hard_and_behavior_scores(
    hard_scores: pd.DataFrame,
    behavior_scores: pd.DataFrame,
    *,
    top_k: int,
    hard_score_semantics: str,
) -> HardBehaviorComparison:
    """Compare an explicitly named hard score with the shadow selector scalar."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("hard-versus-behavior top_k must be a positive integer.")
    if not isinstance(hard_score_semantics, str) or not hard_score_semantics.strip():
        raise ValueError("hard_score_semantics must explicitly name the compared hard score.")
    prediction_provenance = {"prediction_run_id", "prediction_source_sha256"}
    _require_columns(
        hard_scores,
        {"id", "selection_view_id", "hard_score", *prediction_provenance},
        context="hard scores",
    )
    _require_columns(
        behavior_scores,
        {
            "id",
            "selection_view_id",
            "behavior_score",
            "protocol_id",
            "protocol_source_sha256",
            "normalization_source_rows_sha256",
            *prediction_provenance,
        },
        context="behavior scores",
    )
    _require_unique_candidate_views(hard_scores, context="hard scores")
    _require_unique_candidate_views(behavior_scores, context="behavior scores")
    hard_views = set(hard_scores["selection_view_id"].astype(str))
    behavior_views = set(behavior_scores["selection_view_id"].astype(str))
    if hard_views != behavior_views:
        raise ValueError(
            "hard and behavior selection views disagree: "
            f"missing={sorted(behavior_views - hard_views)}, extra={sorted(hard_views - behavior_views)}."
        )

    summaries: list[dict[str, object]] = []
    details: list[pd.DataFrame] = []
    for view_id in sorted(behavior_views):
        hard_view = hard_scores.loc[hard_scores["selection_view_id"].astype(str).eq(view_id)].copy()
        behavior_view = behavior_scores.loc[behavior_scores["selection_view_id"].astype(str).eq(view_id)].copy()
        hard_ids = set(hard_view["id"].astype(str))
        behavior_ids = set(behavior_view["id"].astype(str))
        if hard_ids != behavior_ids:
            raise ValueError(
                f"hard and behavior candidate ids disagree for {view_id!r}: "
                f"missing={sorted(behavior_ids - hard_ids)}, extra={sorted(hard_ids - behavior_ids)}."
            )
        if len(hard_ids) < top_k:
            raise ValueError(f"selection view {view_id!r} has {len(hard_ids)} candidates, fewer than top_k={top_k}.")
        detail = _merge_ranked(hard_view, behavior_view, top_k=top_k)
        detail["hard_score_semantics"] = hard_score_semantics.strip()
        detail["ranking_method"] = "descending_score_then_ascending_candidate_id"
        detail["tie_semantics"] = "ordinal_rank_with_id_tiebreak"
        details.append(detail)
        hard_top = set(detail.loc[detail["hard_selected"], "id"].astype(str))
        behavior_top = set(detail.loc[detail["behavior_selected"], "id"].astype(str))
        summaries.append(
            {
                "selection_view_id": view_id,
                "candidate_count": len(detail),
                "raw_top_k": top_k,
                "raw_top_k_overlap": len(hard_top & behavior_top),
                "raw_top_k_union": len(hard_top | behavior_top),
                "hard_behavior_spearman": _spearman(detail["hard_score"], detail["behavior_score"]),
                "median_absolute_rank_shift": float(detail["rank_shift_behavior_minus_hard"].abs().median()),
                "maximum_absolute_rank_shift": int(detail["rank_shift_behavior_minus_hard"].abs().max()),
                "hard_score_semantics": hard_score_semantics.strip(),
                "ranking_method": "descending_score_then_ascending_candidate_id",
                "tie_semantics": "ordinal_rank_with_id_tiebreak",
                "protocol_id": _single_text(detail, "protocol_id"),
                "protocol_source_sha256": _single_text(detail, "protocol_source_sha256"),
                "normalization_source_rows_sha256": _single_text(detail, "normalization_source_rows_sha256"),
                "prediction_run_id": _single_text(detail, "prediction_run_id"),
                "prediction_source_sha256": _single_text(detail, "prediction_source_sha256"),
                "evidence_role": "fixed_prediction_raw_candidate_ranking_comparison_no_sequence_allocation",
            }
        )
    summary = pd.DataFrame.from_records(summaries).sort_values("selection_view_id", kind="mergesort")
    summary["hard_behavior_spearman"] = pd.array(summary["hard_behavior_spearman"], dtype="Float64")
    detail = pd.concat(details, ignore_index=True).sort_values(
        ["selection_view_id", "behavior_rank", "id"], kind="mergesort"
    )
    return HardBehaviorComparison(summary=summary.reset_index(drop=True), detail=detail.reset_index(drop=True))


def build_repeated_behavior_agreement(observed_scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize score disagreement without aggregating labels or choosing a source."""

    required = {
        "candidate_id",
        "reader_experiment_id",
        "selection_view_id",
        "behavior_score",
        "hard_bottleneck_clearance",
        "limiting_coordinate",
        "protocol_id",
        "protocol_source_sha256",
        "normalization_source_rows_sha256",
    }
    _require_columns(observed_scores, required, context="observed behavior scores")
    if observed_scores.duplicated(subset=["candidate_id", "reader_experiment_id", "selection_view_id"]).any():
        raise ValueError("observed behavior scores must contain one row per candidate experiment and view.")
    records: list[dict[str, object]] = []
    for (candidate_id, view_id), rows in observed_scores.groupby(["candidate_id", "selection_view_id"], sort=True):
        experiments = tuple(sorted(rows["reader_experiment_id"].astype(str).unique()))
        if len(experiments) <= 1:
            continue
        behavior = rows["behavior_score"].to_numpy(dtype=float)
        hard = rows["hard_bottleneck_clearance"].to_numpy(dtype=float)
        records.append(
            {
                "candidate_id": str(candidate_id),
                "selection_view_id": str(view_id),
                "experiment_count": len(experiments),
                "reader_experiment_ids": ",".join(experiments),
                "behavior_score_min": float(np.min(behavior)),
                "behavior_score_max": float(np.max(behavior)),
                "behavior_score_range": float(np.ptp(behavior)),
                "hard_bottleneck_min": float(np.min(hard)),
                "hard_bottleneck_max": float(np.max(hard)),
                "hard_bottleneck_range": float(np.ptp(hard)),
                "limiting_coordinates": ",".join(sorted(rows["limiting_coordinate"].astype(str).unique())),
                "protocol_id": _single_text(rows, "protocol_id"),
                "protocol_source_sha256": _single_text(rows, "protocol_source_sha256"),
                "normalization_source_rows_sha256": _single_text(rows, "normalization_source_rows_sha256"),
                "evidence_role": "repeat_agreement_only_no_label_aggregation_or_source_choice",
            }
        )
    columns = [
        "candidate_id",
        "selection_view_id",
        "experiment_count",
        "reader_experiment_ids",
        "behavior_score_min",
        "behavior_score_max",
        "behavior_score_range",
        "hard_bottleneck_min",
        "hard_bottleneck_max",
        "hard_bottleneck_range",
        "limiting_coordinates",
        "protocol_id",
        "protocol_source_sha256",
        "normalization_source_rows_sha256",
        "evidence_role",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def _merge_ranked(hard: pd.DataFrame, behavior: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    hard_ranked = _ranked(hard, score_column="hard_score", rank_column="hard_rank")
    behavior_ranked = _ranked(behavior, score_column="behavior_score", rank_column="behavior_rank")
    merge_keys = ["id", "selection_view_id", "prediction_run_id", "prediction_source_sha256"]
    detail = hard_ranked.loc[:, [*merge_keys, "hard_score", "hard_rank"]].merge(
        behavior_ranked.loc[
            :,
            [
                "id",
                "selection_view_id",
                "behavior_score",
                "behavior_rank",
                "protocol_id",
                "protocol_source_sha256",
                "normalization_source_rows_sha256",
                "prediction_run_id",
                "prediction_source_sha256",
            ],
        ],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    detail["hard_selected"] = detail["hard_rank"].astype(int).le(top_k)
    detail["behavior_selected"] = detail["behavior_rank"].astype(int).le(top_k)
    detail["rank_shift_behavior_minus_hard"] = detail["behavior_rank"] - detail["hard_rank"]
    return detail


def _ranked(frame: pd.DataFrame, *, score_column: str, rank_column: str) -> pd.DataFrame:
    if not np.isfinite(frame[score_column].to_numpy(dtype=float)).all():
        raise ValueError(f"{score_column} must be finite for rank comparison.")
    result = frame.assign(id=frame["id"].astype(str)).sort_values(
        [score_column, "id"], ascending=[False, True], kind="mergesort"
    )
    result[rank_column] = np.arange(1, len(result) + 1, dtype=int)
    return result


def _require_unique_candidate_views(frame: pd.DataFrame, *, context: str) -> None:
    if frame.duplicated(subset=["id", "selection_view_id"]).any():
        raise ValueError(f"{context} must contain one row per candidate and selection view.")


def _require_columns(frame: pd.DataFrame, required: set[str], *, context: str) -> None:
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"{context} missing columns: {missing}")


def _single_text(frame: pd.DataFrame, column: str) -> str:
    values = tuple(frame[column].astype(str).unique())
    if len(values) != 1:
        raise ValueError(f"behavior comparison requires one {column}; observed {values}.")
    return values[0]


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return None
    return float(spearmanr(left.to_numpy(dtype=float), right.to_numpy(dtype=float)).statistic)


__all__ = ["HardBehaviorComparison", "build_repeated_behavior_agreement", "compare_hard_and_behavior_scores"]
