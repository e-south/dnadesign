"""Feature-term enrichment over persisted plan-margin tail groups."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pyarrow as pa

from ..contracts.errors import ContractViolationError
from .categorical_enrichment import (
    CategoricalEnrichmentConfig,
    CategoricalEnrichmentGroup,
    categorical_enrichment_rows,
)
from .rank_association import RankAssociationConfig, rank_association_rows
from .table_contracts import require_columns, require_sequence, string_values

CONTRACT_NAME = "plan_margin_feature_enrichment"


@dataclass(frozen=True, slots=True)
class PlanMarginFeatureEnrichmentArtifacts:
    """Tables and audit stats emitted by the plan-margin feature builder."""

    enrichment_table: pa.Table
    rank_tests_table: pa.Table
    stats: dict[str, object]


def _unique_native_parent_ids(scores_table: pa.Table) -> list[str]:
    require_columns(scores_table, ["native_parent_id"], contract_name=f"{CONTRACT_NAME} scores")
    parent_ids = [str(value or "").strip() for value in scores_table["native_parent_id"].to_pylist()]
    blank = [index for index, value in enumerate(parent_ids) if not value]
    if blank:
        raise ContractViolationError(f"{CONTRACT_NAME} scores contain blank native_parent_id values")
    duplicates = sorted(parent_id for parent_id, count in Counter(parent_ids).items() if count > 1)
    if duplicates:
        raise ContractViolationError(
            f"{CONTRACT_NAME} scores contain duplicate native_parent_id values: {duplicates[:5]}"
        )
    return parent_ids


def _tail_groups(tail_membership_table: pa.Table, *, universe: set[str]) -> list[CategoricalEnrichmentGroup]:
    required = ["native_parent_id", "plan", "threshold", "tail_mode"]
    require_columns(tail_membership_table, required, contract_name=f"{CONTRACT_NAME} tail_membership")
    members_by_group: dict[tuple[str, float, str], set[str]] = defaultdict(set)
    outside: set[str] = set()
    for row in tail_membership_table.to_pylist():
        parent_id = str(row.get("native_parent_id") or "").strip()
        if not parent_id:
            raise ContractViolationError(f"{CONTRACT_NAME} tail_membership contains blank native_parent_id")
        if parent_id not in universe:
            outside.add(parent_id)
            continue
        group_key = (
            str(row.get("plan") or "").strip(),
            float(row.get("threshold")),
            str(row.get("tail_mode") or "").strip(),
        )
        if not group_key[0] or not group_key[2]:
            raise ContractViolationError(f"{CONTRACT_NAME} tail_membership contains blank plan or tail_mode")
        members_by_group[group_key].add(parent_id)
    if outside:
        raise ContractViolationError(
            f"{CONTRACT_NAME} tail_membership contains parent ids outside score universe: {sorted(outside)[:5]}"
        )
    if not members_by_group:
        raise ContractViolationError(f"{CONTRACT_NAME} matched no tail-membership groups")
    return [
        CategoricalEnrichmentGroup(
            labels={"plan": plan, "threshold": threshold, "tail_mode": tail_mode},
            members=frozenset(members),
        )
        for (plan, threshold, tail_mode), members in sorted(members_by_group.items())
    ]


def _coerce_feature_metadata(
    *,
    existing: dict[str, object],
    column: str,
    feature_id: str,
    raw_value: object,
) -> None:
    value = str(raw_value or "").strip()
    if not value:
        raise ContractViolationError(f"{CONTRACT_NAME} feature {feature_id!r} has blank {column!r}")
    previous = existing.setdefault(column, value)
    if previous != value:
        raise ContractViolationError(
            f"{CONTRACT_NAME} feature {feature_id!r} maps to multiple {column!r} values: {previous!r}, {value!r}"
        )


def _feature_membership(
    feature_table: pa.Table,
    *,
    universe: set[str],
    subject_column: str,
    feature_id_column: str,
    feature_label_column: str,
    feature_namespace_column: str | None,
    namespace_filter: str | None,
    exclude_label_prefixes: Iterable[str],
    source_metadata_columns: Iterable[str],
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, str], dict[str, dict[str, object]], dict[str, object]]:
    if (feature_namespace_column is None) != (namespace_filter is None):
        raise ContractViolationError(
            f"{CONTRACT_NAME} requires feature_namespace_column and namespace_filter to be configured together"
        )
    required_columns = [
        subject_column,
        feature_id_column,
        feature_label_column,
        *(source_metadata_columns or ()),
    ]
    if feature_namespace_column is not None:
        required_columns.append(feature_namespace_column)
    require_columns(
        feature_table,
        list(dict.fromkeys(required_columns)),
        contract_name=f"{CONTRACT_NAME} feature_membership",
    )

    features_by_subject: dict[str, set[str]] = defaultdict(set)
    feature_labels: dict[str, str] = {}
    feature_metadata: dict[str, dict[str, object]] = defaultdict(dict)
    source_columns = list(source_metadata_columns or ())
    total_rows = feature_table.num_rows
    namespace_filtered_rows = 0
    label_filtered_rows = 0
    matched_rows = 0
    orphan_rows = 0
    excluded_prefixes = [str(value).casefold() for value in exclude_label_prefixes if str(value).strip()]

    for row in feature_table.to_pylist():
        if feature_namespace_column is not None and namespace_filter is not None:
            namespace = str(row.get(feature_namespace_column) or "").strip()
            if namespace != namespace_filter:
                namespace_filtered_rows += 1
                continue
        subject_id = str(row.get(subject_column) or "").strip()
        feature_id = str(row.get(feature_id_column) or "").strip()
        feature_label = str(row.get(feature_label_column) or "").strip()
        if not subject_id:
            raise ContractViolationError(f"{CONTRACT_NAME} feature_membership contains blank subject ids")
        if not feature_id:
            raise ContractViolationError(f"{CONTRACT_NAME} feature_membership contains blank feature ids")
        if not feature_label:
            raise ContractViolationError(f"{CONTRACT_NAME} feature {feature_id!r} has a blank label")
        if any(feature_label.casefold().startswith(prefix) for prefix in excluded_prefixes):
            label_filtered_rows += 1
            continue
        if subject_id not in universe:
            orphan_rows += 1
            continue
        matched_rows += 1
        features_by_subject[subject_id].add(feature_id)
        previous_label = feature_labels.setdefault(feature_id, feature_label)
        if previous_label != feature_label:
            raise ContractViolationError(
                f"{CONTRACT_NAME} feature {feature_id!r} maps to multiple labels: {previous_label!r}, {feature_label!r}"
            )
        if feature_namespace_column is not None:
            _coerce_feature_metadata(
                existing=feature_metadata[feature_id],
                column="feature_namespace",
                feature_id=feature_id,
                raw_value=row.get(feature_namespace_column),
            )
        for column in source_columns:
            _coerce_feature_metadata(
                existing=feature_metadata[feature_id],
                column=column,
                feature_id=feature_id,
                raw_value=row.get(column),
            )

    if not any(features_by_subject.values()):
        raise ContractViolationError(f"{CONTRACT_NAME} matched no feature memberships in score universe")
    labels_by_id = {feature_id: feature_id for feature_id in feature_labels}
    stats = {
        "feature_membership_rows": total_rows,
        "namespace_filtered_rows": namespace_filtered_rows,
        "label_filtered_rows": label_filtered_rows,
        "matched_feature_membership_rows": matched_rows,
        "orphan_feature_membership_rows": orphan_rows,
        "matched_features": len(feature_labels),
        "feature_namespace_filter": namespace_filter,
        "excluded_label_prefixes": excluded_prefixes,
    }
    return features_by_subject, labels_by_id, feature_labels, feature_metadata, stats


def _map_generic_rows(
    generic_rows: list[dict[str, object]],
    *,
    feature_labels: Mapping[str, str],
    feature_metadata: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for row in generic_rows:
        feature_id = str(row["feature"])
        output_rows.append(
            {
                "feature_id": feature_id,
                "feature_label": feature_labels[feature_id],
                **dict(feature_metadata.get(feature_id, {})),
                "plan": row["plan"],
                "threshold": row["threshold"],
                "tail_mode": row["tail_mode"],
                "n_total_native": row["n_total"],
                "n_tail": row["n_group"],
                "n_feature_total": row["n_feature_total"],
                "n_feature_tail": row["n_feature_group"],
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
                "is_common_feature": row["is_common_feature"],
                "notes": row["notes"],
            }
        )
    return output_rows


def _score_rows_by_subject(
    scores_table: pa.Table,
    *,
    parent_ids: list[str],
    plans: list[str],
) -> dict[str, dict[str, float]]:
    margin_columns = [f"margin_{plan}" for plan in plans]
    require_columns(scores_table, margin_columns, contract_name=f"{CONTRACT_NAME} scores")
    columns = {column: scores_table[column].to_pylist() for column in margin_columns}
    scores_by_subject: dict[str, dict[str, float]] = {}
    for index, parent_id in enumerate(parent_ids):
        scores_by_subject[parent_id] = {plan: float(columns[f"margin_{plan}"][index]) for plan in plans}
    return scores_by_subject


def _plans_from_scores_and_groups(
    scores_table: pa.Table,
    groups: Iterable[CategoricalEnrichmentGroup],
) -> list[str]:
    score_plans = [
        column.removeprefix("margin_") for column in scores_table.column_names if column.startswith("margin_")
    ]
    if not score_plans:
        raise ContractViolationError(f"{CONTRACT_NAME} scores contain no margin_<plan> columns")
    group_plans: list[str] = []
    for group in groups:
        plan = str(group.labels.get("plan") or "").strip()
        if not plan:
            raise ContractViolationError(f"{CONTRACT_NAME} tail groups must include nonblank plan labels")
        if plan not in group_plans:
            group_plans.append(plan)
    missing = sorted(set(group_plans).difference(score_plans))
    if missing:
        raise ContractViolationError(f"{CONTRACT_NAME} tail groups reference plans missing from scores: {missing[:5]}")
    extra = sorted(set(score_plans).difference(group_plans))
    if extra:
        raise ContractViolationError(
            f"{CONTRACT_NAME} scores contain margin plans absent from tail groups: {extra[:5]}"
        )
    return group_plans


def _map_rank_rows(
    generic_rows: list[dict[str, object]],
    *,
    feature_labels: Mapping[str, str],
    feature_metadata: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for row in generic_rows:
        feature_id = str(row["feature"])
        output_rows.append(
            {
                "feature_id": feature_id,
                "feature_label": feature_labels[feature_id],
                **dict(feature_metadata.get(feature_id, {})),
                "plan": row["axis"],
                "n_total_native": row["n_total"],
                "n_with_feature": row["n_with"],
                "n_without_feature": row["n_without"],
                "median_margin_with_feature": row["median_with"],
                "median_margin_without_feature": row["median_without"],
                "u_statistic": row["u_statistic"],
                "auc": row["auc"],
                "rank_biserial": row["rank_biserial"],
                "p_value": row["p_value"],
                "q_value": row["q_value"],
                "p_value_method": row["p_value_method"],
                "p_value_alternative": row["p_value_alternative"],
                "fdr_method": row["fdr_method"],
                "passes_min_support": row["passes_min_feature_support"],
                "is_common_feature": row["is_common_feature"],
                "notes": row["notes"],
            }
        )
    return output_rows


def build_plan_margin_feature_enrichment_artifact(
    *,
    scores_table: pa.Table,
    tail_membership_table: pa.Table,
    feature_table: pa.Table,
    subject_column: str,
    feature_id_column: str,
    feature_label_column: str,
    feature_namespace_column: str | None = None,
    namespace_filter: str | None = None,
    exclude_label_prefixes: Iterable[str] = (),
    source_metadata_columns: Iterable[str] = (),
    min_global_subjects: int = 10,
    min_tail_hits: int = 3,
    rank_test_alternative: str = "greater",
    common_features: Iterable[str] = (),
) -> PlanMarginFeatureEnrichmentArtifacts:
    """Build categorical-feature enrichment over existing plan-margin tails."""

    if min_global_subjects < 1:
        raise ContractViolationError(f"{CONTRACT_NAME} min_global_subjects must be >= 1")
    if min_tail_hits < 1:
        raise ContractViolationError(f"{CONTRACT_NAME} min_tail_hits must be >= 1")
    metadata_columns = string_values(
        source_metadata_columns,
        field_name="source_metadata_columns",
        contract_name=CONTRACT_NAME,
    )
    excluded_label_prefixes = [
        str(value).casefold()
        for value in require_sequence(
            exclude_label_prefixes,
            field_name="exclude_label_prefixes",
            contract_name=CONTRACT_NAME,
        )
        if str(value).strip()
    ]
    parent_ids = _unique_native_parent_ids(scores_table)
    universe = set(parent_ids)
    groups = _tail_groups(tail_membership_table, universe=universe)
    plans = _plans_from_scores_and_groups(scores_table, groups)
    features_by_subject, labels_by_id, display_labels_by_id, feature_metadata, membership_stats = _feature_membership(
        feature_table,
        universe=universe,
        subject_column=subject_column,
        feature_id_column=feature_id_column,
        feature_label_column=feature_label_column,
        feature_namespace_column=feature_namespace_column,
        namespace_filter=namespace_filter,
        exclude_label_prefixes=excluded_label_prefixes,
        source_metadata_columns=metadata_columns,
    )
    generic_rows = categorical_enrichment_rows(
        universe_ids=parent_ids,
        features_by_subject=features_by_subject,
        feature_labels=labels_by_id,
        groups=groups,
        config=CategoricalEnrichmentConfig(
            min_feature_support=min_global_subjects,
            min_group_hits=min_tail_hits,
        ),
        common_features=common_features,
    )
    rank_rows = rank_association_rows(
        universe_ids=parent_ids,
        score_by_subject=_score_rows_by_subject(scores_table, parent_ids=parent_ids, plans=plans),
        features_by_subject=features_by_subject,
        feature_labels=labels_by_id,
        axis_ids=plans,
        config=RankAssociationConfig(
            min_feature_support=min_global_subjects,
            alternative=rank_test_alternative,
        ),
        common_features=common_features,
    )
    output_rows = _map_generic_rows(
        generic_rows,
        feature_labels=display_labels_by_id,
        feature_metadata=feature_metadata,
    )
    output_rank_rows = _map_rank_rows(
        rank_rows,
        feature_labels=display_labels_by_id,
        feature_metadata=feature_metadata,
    )
    return PlanMarginFeatureEnrichmentArtifacts(
        enrichment_table=pa.Table.from_pylist(output_rows),
        rank_tests_table=pa.Table.from_pylist(output_rank_rows),
        stats={
            "native_rows": len(parent_ids),
            "tail_groups": len(groups),
            "rank_test_rows": len(output_rank_rows),
            "enrichment_rows": len(output_rows),
            "min_global_subjects": min_global_subjects,
            "min_tail_hits": min_tail_hits,
            "rank_test_alternative": rank_test_alternative,
            "source_metadata_columns": metadata_columns,
            **membership_stats,
        },
    )
