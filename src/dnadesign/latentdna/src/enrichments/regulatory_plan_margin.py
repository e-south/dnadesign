"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/enrichments/regulatory_plan_margin.py

Regulatory association enrichment over synthetic-plan latent margins.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
import pyarrow as pa

from ..contracts.errors import ContractViolationError
from ..geometry.preprocessing import standardize_and_l2_normalize, try_l2_normalize_vector
from .categorical_enrichment import (
    CategoricalEnrichmentConfig,
    CategoricalEnrichmentGroup,
    categorical_enrichment_rows,
)
from .rank_association import RankAssociationConfig, rank_association_rows
from .regulatory_plan_margin_contracts import (
    EPS,
    RegulatoryPlanMarginArtifacts,
    coerce_centroid_groups,
    string_values,
    validate_tail_modes,
    validate_thresholds,
)
from .table_contracts import filter_indices, require_columns


def _plan_centroids(
    normalized: np.ndarray,
    rows: list[dict[str, object]],
    *,
    cohort_column: str,
    centroid_groups: dict[str, set[str]],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    centroids: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for plan, values in centroid_groups.items():
        indices = [index for index, row in enumerate(rows) if str(row.get(cohort_column) or "").strip() in values]
        if not indices:
            raise ContractViolationError(
                f"native_regulator_plan_margin_enrichment centroid group {plan!r} matched no rows on {cohort_column!r}"
            )
        centroid = try_l2_normalize_vector(np.asarray(normalized[indices].mean(axis=0), dtype=np.float32))
        if centroid is None:
            raise ContractViolationError(
                f"native_regulator_plan_margin_enrichment centroid group {plan!r} is degenerate"
            )
        centroids[plan] = centroid
        counts[plan] = len(indices)
    return centroids, counts


def _unique_native_parent_ids(
    native_rows: list[dict[str, object]],
    *,
    native_parent_column: str,
) -> list[str]:
    parent_ids: list[str] = []
    for row in native_rows:
        value = str(row.get(native_parent_column) or "").strip()
        if not value:
            raise ContractViolationError(
                f"native_regulator_plan_margin_enrichment encountered a native row without {native_parent_column!r}"
            )
        parent_ids.append(value)
    duplicates = sorted(parent_id for parent_id, count in Counter(parent_ids).items() if count > 1)
    if duplicates:
        raise ContractViolationError(
            f"native_regulator_plan_margin_enrichment duplicate native parent ids: {duplicates[:5]}"
        )
    return parent_ids


def _regulator_membership(
    relations_table: pa.Table,
    *,
    native_parent_ids: set[str],
    relation_key: str,
    regulator_column: str,
    required_relation_columns: Iterable[str],
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, object]]:
    extra_required_columns = string_values(
        required_relation_columns,
        field_name="regulatory_interactions.required_columns",
    )
    required_columns = list(
        dict.fromkeys(
            [
                relation_key,
                regulator_column,
                *extra_required_columns,
            ]
        )
    )
    require_columns(
        relations_table,
        required_columns,
        contract_name="native_regulator_plan_margin_enrichment regulatory_interactions",
    )
    labels_by_normalized: dict[str, str] = {}
    regulators_by_parent: dict[str, set[str]] = defaultdict(set)
    relation_rows = relations_table.to_pylist()
    matched_relation_rows = 0
    orphan_relation_rows = 0
    for relation in relation_rows:
        parent_id = str(relation.get(relation_key) or "").strip()
        regulator = str(relation.get(regulator_column) or "").strip()
        if not parent_id or not regulator:
            continue
        if parent_id not in native_parent_ids:
            orphan_relation_rows += 1
            continue
        matched_relation_rows += 1
        regulator_key = regulator.casefold()
        labels_by_normalized.setdefault(regulator_key, regulator)
        regulators_by_parent[parent_id].add(regulator_key)
    if not any(regulators_by_parent.values()):
        raise ContractViolationError(
            "native_regulator_plan_margin_enrichment matched no regulatory associations in native row scope"
        )
    stats = {
        "regulatory_interaction_rows": relations_table.num_rows,
        "matched_relation_rows": matched_relation_rows,
        "orphan_relation_rows": orphan_relation_rows,
        "matched_regulators": len(labels_by_normalized),
    }
    return regulators_by_parent, labels_by_normalized, stats


def _native_score_rows(
    native_rows: list[dict[str, object]],
    native_vectors: np.ndarray,
    centroids: dict[str, np.ndarray],
    *,
    plan_order: list[str],
    view_id: str,
    parent_ids: list[str],
    regulators_by_parent: dict[str, set[str]],
    metadata_columns: list[str],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    similarity_by_plan = {plan: np.asarray(native_vectors @ centroids[plan], dtype=np.float32) for plan in plan_order}
    similarity_matrix = np.column_stack([similarity_by_plan[plan] for plan in plan_order])
    nearest_indices = np.argmax(similarity_matrix, axis=1)
    score_rows: list[dict[str, object]] = []
    margin_by_plan: dict[str, np.ndarray] = {}
    for plan_index, plan in enumerate(plan_order):
        other = np.delete(similarity_matrix, plan_index, axis=1)
        margin_by_plan[plan] = np.asarray(similarity_matrix[:, plan_index] - np.max(other, axis=1), dtype=np.float32)
    for index, (row, parent_id) in enumerate(zip(native_rows, parent_ids, strict=True)):
        similarities = similarity_matrix[index]
        maximum = float(np.max(similarities))
        tie_count = int(np.count_nonzero(np.isclose(similarities, maximum, atol=EPS, rtol=0.0)))
        score_rows.append(
            {
                "native_parent_id": parent_id,
                **{column: row.get(column) for column in metadata_columns},
                "embedding_view": view_id,
                **{f"sim_{plan}": float(similarity_by_plan[plan][index]) for plan in plan_order},
                **{f"margin_{plan}": float(margin_by_plan[plan][index]) for plan in plan_order},
                "nearest_plan": plan_order[int(nearest_indices[index])],
                "nearest_plan_tie_count": tie_count,
                "regulator_degree": len(regulators_by_parent.get(parent_id, set())),
            }
        )
    return score_rows, margin_by_plan


def _tail_membership_rows(
    score_rows: list[dict[str, object]],
    margin_by_plan: dict[str, np.ndarray],
    *,
    plan_order: list[str],
    thresholds: list[float],
    tail_modes: list[str],
) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    parent_ids = [str(row["native_parent_id"]) for row in score_rows]
    nearest_plans = [str(row["nearest_plan"]) for row in score_rows]
    row_count = len(score_rows)
    for plan in plan_order:
        margins = margin_by_plan[plan]
        ranked_indices = sorted(range(row_count), key=lambda index: (-float(margins[index]), parent_ids[index]))
        ranks = {index: rank for rank, index in enumerate(ranked_indices, start=1)}
        for threshold in thresholds:
            cutoff = max(1, int(math.ceil(row_count * threshold)))
            candidate_indices = ranked_indices[:cutoff]
            for mode in tail_modes:
                if mode == "margin_top_quantile_nearest_plan_only":
                    selected_indices = [index for index in candidate_indices if nearest_plans[index] == plan]
                else:
                    selected_indices = candidate_indices
                for index in selected_indices:
                    output_rows.append(
                        {
                            "native_parent_id": parent_ids[index],
                            "plan": plan,
                            "threshold": float(threshold),
                            "tail_mode": mode,
                            "rank": int(ranks[index]),
                            "tail_cutoff_rank": int(cutoff),
                            "margin": float(margins[index]),
                            "nearest_plan": nearest_plans[index],
                        }
                    )
    return output_rows


def _enrichment_rows(
    *,
    plan_order: list[str],
    native_parent_ids: list[str],
    regulators_by_parent: dict[str, set[str]],
    labels_by_normalized: dict[str, str],
    tail_rows: list[dict[str, object]],
    thresholds: list[float],
    tail_modes: list[str],
    min_global_promoters: int,
    min_tail_hits: int,
    common_regulators: set[str],
) -> list[dict[str, object]]:
    tail_sets: dict[tuple[str, float, str], set[str]] = {}
    for plan in plan_order:
        for threshold in thresholds:
            for tail_mode in tail_modes:
                tail_sets[(plan, threshold, tail_mode)] = set()
    for row in tail_rows:
        key = (str(row["plan"]), float(row["threshold"]), str(row["tail_mode"]))
        tail_sets.setdefault(key, set()).add(str(row["native_parent_id"]))

    groups = [
        CategoricalEnrichmentGroup(
            labels={
                "plan": plan,
                "threshold": float(threshold),
                "tail_mode": tail_mode,
            },
            members=frozenset(tail_sets[(plan, threshold, tail_mode)]),
        )
        for plan in plan_order
        for threshold in thresholds
        for tail_mode in tail_modes
    ]
    generic_rows = categorical_enrichment_rows(
        universe_ids=native_parent_ids,
        features_by_subject=regulators_by_parent,
        feature_labels=labels_by_normalized,
        groups=groups,
        config=CategoricalEnrichmentConfig(
            min_feature_support=min_global_promoters,
            min_group_hits=min_tail_hits,
        ),
        common_features=common_regulators,
    )
    output_rows: list[dict[str, object]] = []
    for row in generic_rows:
        notes = str(row["notes"])
        notes = notes.replace("below_min_feature_support", "below_min_support")
        notes = notes.replace("below_min_group_hits", "below_min_tail_hits")
        notes = notes.replace("common_feature", "common_regulator")
        output_rows.append(
            {
                "regulator_abbrev": row["feature"],
                "plan": row["plan"],
                "threshold": row["threshold"],
                "tail_mode": row["tail_mode"],
                "n_total_native": row["n_total"],
                "n_tail": row["n_group"],
                "n_regulator_total": row["n_feature_total"],
                "n_regulator_tail": row["n_feature_group"],
                "tail_fraction": row["group_fraction"],
                "background_fraction": row["background_fraction"],
                "enrichment_ratio": row["enrichment_ratio"],
                "odds_ratio": row["odds_ratio"],
                "p_value": row["p_value"],
                "q_value": row["q_value"],
                "p_value_method": row["p_value_method"],
                "fdr_method": row["fdr_method"],
                "passes_min_support": row["passes_min_feature_support"],
                "passes_min_tail_hits": row["passes_min_group_hits"],
                "is_common_regulator": row["is_common_feature"],
                "notes": notes,
            }
        )
    return output_rows


def _margin_scores_by_parent(
    *,
    native_parent_ids: list[str],
    margin_by_plan: dict[str, np.ndarray],
    plan_order: list[str],
) -> dict[str, dict[str, float]]:
    return {
        parent_id: {plan: float(margin_by_plan[plan][index]) for plan in plan_order}
        for index, parent_id in enumerate(native_parent_ids)
    }


def _rank_test_rows(
    *,
    plan_order: list[str],
    native_parent_ids: list[str],
    margin_by_plan: dict[str, np.ndarray],
    regulators_by_parent: dict[str, set[str]],
    labels_by_normalized: dict[str, str],
    min_global_promoters: int,
    common_regulators: set[str],
    alternative: str,
) -> list[dict[str, object]]:
    generic_rows = rank_association_rows(
        universe_ids=native_parent_ids,
        score_by_subject=_margin_scores_by_parent(
            native_parent_ids=native_parent_ids,
            margin_by_plan=margin_by_plan,
            plan_order=plan_order,
        ),
        features_by_subject=regulators_by_parent,
        feature_labels=labels_by_normalized,
        axis_ids=plan_order,
        config=RankAssociationConfig(
            min_feature_support=min_global_promoters,
            alternative=alternative,
        ),
        common_features=common_regulators,
    )
    output_rows: list[dict[str, object]] = []
    for row in generic_rows:
        notes = str(row["notes"])
        notes = notes.replace("below_min_feature_support", "below_min_support")
        notes = notes.replace("common_feature", "common_regulator")
        output_rows.append(
            {
                "regulator_abbrev": row["feature"],
                "plan": row["axis"],
                "n_total_native": row["n_total"],
                "n_with_regulator": row["n_with"],
                "n_without_regulator": row["n_without"],
                "median_margin_with_regulator": row["median_with"],
                "median_margin_without_regulator": row["median_without"],
                "u_statistic": row["u_statistic"],
                "auc": row["auc"],
                "rank_biserial": row["rank_biserial"],
                "p_value": row["p_value"],
                "q_value": row["q_value"],
                "p_value_method": row["p_value_method"],
                "p_value_alternative": row["p_value_alternative"],
                "fdr_method": row["fdr_method"],
                "passes_min_support": row["passes_min_feature_support"],
                "is_common_regulator": row["is_common_feature"],
                "notes": notes,
            }
        )
    return output_rows


def build_regulatory_plan_margin_artifacts(
    *,
    matrix: np.ndarray,
    rows_table: pa.Table,
    relations_table: pa.Table,
    view_id: str,
    cohort_column: str,
    centroid_groups: dict[str, object],
    native_filter: dict[str, object],
    native_parent_column: str,
    relation_key: str,
    regulator_column: str,
    thresholds: Iterable[object],
    tail_modes: Iterable[object],
    min_global_promoters: int,
    min_tail_hits: int,
    common_regulators: Iterable[object],
    plan_order: Iterable[object] | None = None,
    native_metadata_columns: Iterable[object] = (),
    required_relation_columns: Iterable[object] = (),
    fdr_method: str = "benjamini_hochberg",
    rank_test_alternative: str = "greater",
    expected_output_rows: int | None = None,
) -> RegulatoryPlanMarginArtifacts:
    """Build plan-margin score, tail-membership, and regulator-enrichment tables."""

    if fdr_method != "benjamini_hochberg":
        raise ContractViolationError(
            "native_regulator_plan_margin_enrichment only supports fdr_method='benjamini_hochberg'"
        )
    if min_global_promoters < 1:
        raise ContractViolationError("native_regulator_plan_margin_enrichment min_global_promoters must be >= 1")
    if min_tail_hits < 1:
        raise ContractViolationError("native_regulator_plan_margin_enrichment min_tail_hits must be >= 1")
    matrix_array = np.asarray(matrix, dtype=np.float32)
    if matrix_array.ndim != 2 or matrix_array.shape[0] != rows_table.num_rows:
        raise ContractViolationError("native_regulator_plan_margin_enrichment view matrix and rows are misaligned")
    metadata_columns = string_values(native_metadata_columns, field_name="native_metadata_columns")
    require_columns(
        rows_table,
        [cohort_column, native_parent_column, *metadata_columns],
        contract_name="native_regulator_plan_margin_enrichment view rows",
    )
    thresholds_list = validate_thresholds(thresholds)
    tail_modes_list = validate_tail_modes(tail_modes)
    ordered_plans, groups = coerce_centroid_groups(centroid_groups, plan_order=plan_order)
    normalized = standardize_and_l2_normalize(
        matrix_array,
        nonfinite_policy="error",
        zero_variance_policy="drop_or_zero",
        zero_row_policy="zero",
    )
    rows = rows_table.to_pylist()
    centroids, centroid_counts = _plan_centroids(
        normalized,
        rows,
        cohort_column=cohort_column,
        centroid_groups=groups,
    )
    native_indices = filter_indices(
        rows_table,
        native_filter,
        contract_name="native_regulator_plan_margin_enrichment native_filter",
    )
    if not native_indices:
        raise ContractViolationError("native_regulator_plan_margin_enrichment native_filter matched no rows")
    native_rows = [rows[index] for index in native_indices]
    parent_ids = _unique_native_parent_ids(native_rows, native_parent_column=native_parent_column)
    if expected_output_rows is not None and len(parent_ids) != expected_output_rows:
        raise ContractViolationError(
            "native_regulator_plan_margin_enrichment expected_output_rows mismatch: "
            f"expected {expected_output_rows}, observed {len(parent_ids)}"
        )
    regulators_by_parent, labels_by_normalized, relation_stats = _regulator_membership(
        relations_table,
        native_parent_ids=set(parent_ids),
        relation_key=relation_key,
        regulator_column=regulator_column,
        required_relation_columns=required_relation_columns,
    )
    native_vectors = np.asarray(normalized[np.asarray(native_indices, dtype=np.int64)], dtype=np.float32)
    score_rows, margin_by_plan = _native_score_rows(
        native_rows,
        native_vectors,
        centroids,
        plan_order=ordered_plans,
        view_id=view_id,
        parent_ids=parent_ids,
        regulators_by_parent=regulators_by_parent,
        metadata_columns=metadata_columns,
    )
    tail_rows = _tail_membership_rows(
        score_rows,
        margin_by_plan,
        plan_order=ordered_plans,
        thresholds=thresholds_list,
        tail_modes=tail_modes_list,
    )
    normalized_common_regulators = {
        value.casefold() for value in string_values(common_regulators, field_name="common_regulators")
    }
    enrichment_rows = _enrichment_rows(
        plan_order=ordered_plans,
        native_parent_ids=parent_ids,
        regulators_by_parent=regulators_by_parent,
        labels_by_normalized=labels_by_normalized,
        tail_rows=tail_rows,
        thresholds=thresholds_list,
        tail_modes=tail_modes_list,
        min_global_promoters=min_global_promoters,
        min_tail_hits=min_tail_hits,
        common_regulators=normalized_common_regulators,
    )
    rank_test_rows = _rank_test_rows(
        plan_order=ordered_plans,
        native_parent_ids=parent_ids,
        margin_by_plan=margin_by_plan,
        regulators_by_parent=regulators_by_parent,
        labels_by_normalized=labels_by_normalized,
        min_global_promoters=min_global_promoters,
        common_regulators=normalized_common_regulators,
        alternative=rank_test_alternative,
    )
    return RegulatoryPlanMarginArtifacts(
        scores_table=pa.Table.from_pylist(score_rows),
        tail_membership_table=pa.Table.from_pylist(tail_rows),
        rank_tests_table=pa.Table.from_pylist(rank_test_rows),
        enrichment_table=pa.Table.from_pylist(enrichment_rows),
        stats={
            "view_id": view_id,
            "input_rows": rows_table.num_rows,
            "native_rows": len(parent_ids),
            "expected_output_rows": expected_output_rows,
            "centroid_counts": centroid_counts,
            "plan_order": ordered_plans,
            "native_metadata_columns": metadata_columns,
            "thresholds": thresholds_list,
            "tail_modes": tail_modes_list,
            "min_global_promoters": min_global_promoters,
            "min_tail_hits": min_tail_hits,
            "fdr_method": fdr_method,
            "rank_test_alternative": rank_test_alternative,
            "score_rows": len(score_rows),
            "tail_membership_rows": len(tail_rows),
            "rank_test_rows": len(rank_test_rows),
            "enrichment_rows": len(enrichment_rows),
            **relation_stats,
        },
    )
