"""Pre-assay scalar builders for promoter representation triage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts.errors import ContractViolationError
from ..geometry.cohorts import (
    aligned_cohort_distance_vectors,
    balanced_group_indices,
    bootstrap_ci,
    centroid_map,
    group_indices,
    ordinal_gap_and_distance_vectors,
    resample_groups,
    separation_ratio_from_groups,
)
from ..geometry.preprocessing import try_l2_normalize_vector
from ..io.json_io import read_json
from ..io.parquet_io import write_table
from ..workspaces.loader import WorkspaceContext
from .common import (
    BuiltScalarArtifact,
    ScalarInputRef,
    _candidate_descriptor_from_view,
    _cosine_distance_upper,
    _effective_rank,
    _load_sig35_order,
    _load_view_scope_table,
    _metric_row,
    _normalized_geometry_rows,
    _optional_param,
    _pearson_correlation,
    _reducer_summary_path,
    _require_param,
    _sig35_global_statistics,
    _sig35_mean_statistic_from_outer_groups,
    _sig35_mean_statistic_within_groups,
    _sig35_statistics_from_groups,
    _spearman_correlation,
)

ScalarBuilderResult = tuple[pa.Table, list[ScalarInputRef], dict[str, object]]
ScalarTableBuilder = Callable[[WorkspaceContext, dict[str, Any]], ScalarBuilderResult]


_REFERENCE_GROUP_COLUMN_LABELS = {
    "source_family": "Src",
    "selection_basis": "Basis",
    "promoter_standard__collection_id": "Std",
}

_REFERENCE_GROUP_VALUE_LABELS = {
    "anderson_igem": "Anderson iGEM",
    "archive_backed_insert": "Archive Insert",
    "construct_derived": "Construct-Derived",
    "legacy_construct_seed": "Legacy Seed",
    "legacy_reference_control": "Legacy Reference",
    "native_source_length": "Native Length",
    "reference_source": "Reference Source",
    "sfxi_archive": "SFXI Archive",
    "sigma_site_pair_midpoint": "Sigma Midpoint",
    "t7_w_collection": "T7 W collection",
    "template_window_center": "Template Window",
}

_REFERENCE_GROUP_METRIC_LABELS = {
    "reference_group_size": "Reference group size",
    "reference_group_pairwise_cosine_distance_median": "Reference group median distance",
    "reference_group_pairwise_cosine_distance_iqr": "Reference group distance IQR",
}


@dataclass(frozen=True, slots=True)
class _CandidateSample:
    descriptor: dict[str, object]
    matrix: np.ndarray
    rows: list[dict[str, object]]
    inputs: list[ScalarInputRef]


def _load_candidate_sample(
    context: WorkspaceContext,
    candidate: dict[str, Any],
) -> _CandidateSample:
    view_id = str(_require_param(candidate, "view_id"))
    sample_id = _optional_param(candidate, "sample_id", default=None)
    descriptor = _candidate_descriptor_from_view(context, view_id=view_id)
    matrix, rows, inputs = _load_view_scope_table(context, view_id=view_id, sample_id=sample_id)
    return _CandidateSample(
        descriptor=descriptor,
        matrix=matrix,
        rows=rows,
        inputs=inputs,
    )


def _reference_group_label(value: object) -> str:
    text = " ".join(str(value or "").replace("__", " ").replace("_", " ").split()).strip()
    if not text:
        return ""
    words = []
    for word in text.split(" "):
        lowered = word.lower()
        if lowered in {"igem", "t7", "w"}:
            words.append(word.upper())
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)


def _reference_group_panel_title(*, metric_id: str, group_column: str, group_value: str) -> str:
    metric_label = _REFERENCE_GROUP_METRIC_LABELS.get(metric_id, _reference_group_label(metric_id))
    column_label = _REFERENCE_GROUP_COLUMN_LABELS.get(group_column, _reference_group_label(group_column))
    value_label = _REFERENCE_GROUP_VALUE_LABELS.get(group_value, _reference_group_label(group_value))
    return f"{metric_label}\n{column_label}: {value_label}"


def _load_scalar_rows(
    context: WorkspaceContext,
    *,
    scalar_id: str,
) -> tuple[list[dict[str, object]], list[ScalarInputRef]]:
    path = context.output_root / "scalars" / scalar_id / "table.parquet"
    if not path.is_file():
        raise ContractViolationError(f"pre-assay scalar source is missing: {scalar_id}")
    table = pq.read_table(path)
    return table.to_pylist(), [ScalarInputRef(kind="scalar_table", artifact_id=scalar_id, path=path)]


def _rows_by_candidate_and_metric(rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    grouped: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        metric_id = str(row.get("metric_id") or "").strip()
        if not candidate_id or not metric_id:
            continue
        grouped.setdefault(candidate_id, {})[metric_id] = row
    return grouped


def _representation_health_summary_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    candidates = [dict(value) for value in _require_param(params, "candidates")]
    collapse_rules = {
        str(key): float(value) for key, value in dict(_optional_param(params, "collapse_rules", default={})).items()
    }
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    for candidate in candidates:
        candidate_sample = _load_candidate_sample(context, candidate)
        reducer_id = str(_require_param(candidate, "reducer_id"))
        inputs.extend(candidate_sample.inputs)
        reducer_path = _reducer_summary_path(context, reducer_id)
        inputs.append(ScalarInputRef(kind="reducer", artifact_id=reducer_id, path=reducer_path))
        reducer_summary = read_json(reducer_path)
        explained = [float(value) for value in reducer_summary.get("explained_variance_ratio", [])]
        pc1_fraction = float(explained[0]) if explained else float("nan")
        distances = _cosine_distance_upper(candidate_sample.matrix)
        distance_median = float(np.median(distances)) if distances.size else float("nan")
        distance_iqr = (
            float(np.percentile(distances, 75.0) - np.percentile(distances, 25.0)) if distances.size else float("nan")
        )
        effective_rank = _effective_rank(explained)
        failures = sum(
            [
                effective_rank < float(collapse_rules.get("effective_rank_min", 2.0)),
                pc1_fraction > float(collapse_rules.get("pc1_fraction_max", 0.80)),
                distance_iqr < float(collapse_rules.get("pairwise_distance_iqr_min", 0.01)),
            ]
        )
        health_status = "fail" if failures >= 2 else "warn" if failures == 1 else "pass"
        extra = {
            "health_status": health_status,
            "collapse_flag": health_status != "pass",
            "effective_rank_basis": "retained_pca_components",
            "effective_rank_component_count": len([value for value in explained if value > 0.0]),
            "explained_variance_captured": float(sum(explained)),
            "pca_fit_rows": int(reducer_summary.get("fit_rows", 0) or 0),
            "pca_input_dims": int(reducer_summary.get("input_dims", 0) or 0),
            "pca_output_dims": int(reducer_summary.get("output_dims", len(explained)) or len(explained)),
            "pca_fit_scope_kind": str(reducer_summary.get("scope_kind") or ""),
            "pca_fit_scope_id": str(reducer_summary.get("scope_id") or ""),
            "pca_method": str(reducer_summary.get("pca_method") or reducer_summary.get("method") or ""),
        }
        rows.extend(
            [
                _metric_row(
                    descriptor=candidate_sample.descriptor,
                    metric_id="effective_rank",
                    metric_value=effective_rank,
                    extra=extra,
                ),
                _metric_row(
                    descriptor=candidate_sample.descriptor,
                    metric_id="pc1_variance_fraction",
                    metric_value=pc1_fraction,
                    extra=extra,
                ),
                _metric_row(
                    descriptor=candidate_sample.descriptor,
                    metric_id="pairwise_cosine_distance_median",
                    metric_value=distance_median,
                    extra=extra,
                ),
                _metric_row(
                    descriptor=candidate_sample.descriptor,
                    metric_id="pairwise_cosine_distance_iqr",
                    metric_value=distance_iqr,
                    extra=extra,
                ),
            ]
        )
    return pa.Table.from_pylist(rows), inputs, {"candidate_count": len(candidates), "rows": len(rows)}


def _design_structure_summary_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    candidates = [dict(value) for value in _require_param(params, "candidates")]
    bootstrap_iterations = int(_optional_param(params, "bootstrap_iterations", default=200))
    seed = int(_optional_param(params, "seed", default=context.config.defaults.random_seed))
    balance_columns = [
        str(value) for value in _optional_param(params, "balance_columns", default=["sig35_variant", "spacer_length"])
    ]
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    synthetic_design_families = {"background_only", "ethanol", "ciprofloxacin", "ethanol_ciprofloxacin"}
    metric_specs = [
        ("design_family", "design_family_separation_ratio", {"control"}),
        ("design_regulator_composition", "design_regulator_composition_separation_ratio", {"control"}),
        ("sig35_variant", "sig35_variant_separation_ratio", {"control"}),
        ("spacer_length", "spacer_length_separation_ratio", None),
    ]
    for offset, candidate in enumerate(candidates):
        rng = np.random.default_rng(seed + offset)
        candidate_sample = _load_candidate_sample(context, candidate)
        inputs.extend(candidate_sample.inputs)
        normalized = _normalized_geometry_rows(candidate_sample.matrix)
        metric_values: dict[str, float] = {}
        for column, metric_id, exclude_values in metric_specs:
            groups = group_indices(candidate_sample.rows, column=column, exclude_values=exclude_values)
            value = separation_ratio_from_groups(normalized, groups)
            metric_values[metric_id] = value
            ci_lower, ci_upper = bootstrap_ci(
                lambda groups=groups, rng=rng: separation_ratio_from_groups(
                    normalized,
                    resample_groups(groups, rng=rng),
                ),
                iterations=bootstrap_iterations,
            )
            rows.append(
                _metric_row(
                    descriptor=candidate_sample.descriptor,
                    metric_id=metric_id,
                    metric_value=value,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                )
            )

        balanced_groups = balanced_group_indices(
            candidate_sample.rows,
            group_column="design_family",
            balance_columns=balance_columns,
            required_group_values=synthetic_design_families,
            exclude_group_values={"control"},
            rng=rng,
        )
        balanced_value = separation_ratio_from_groups(normalized, balanced_groups)
        metric_values["design_family_balanced_separation_ratio"] = balanced_value
        ci_lower, ci_upper = bootstrap_ci(
            lambda rng=rng: separation_ratio_from_groups(
                normalized,
                balanced_group_indices(
                    candidate_sample.rows,
                    group_column="design_family",
                    balance_columns=balance_columns,
                    required_group_values=synthetic_design_families,
                    exclude_group_values={"control"},
                    rng=rng,
                ),
            ),
            iterations=bootstrap_iterations,
        )
        rows.append(
            _metric_row(
                descriptor=candidate_sample.descriptor,
                metric_id="design_family_balanced_separation_ratio",
                metric_value=balanced_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                extra={
                    "spacer_length_dominates_design_family": bool(
                        metric_values.get("spacer_length_separation_ratio", float("-inf")) > balanced_value
                    )
                },
            )
        )
    return pa.Table.from_pylist(rows), inputs, {"candidate_count": len(candidates), "rows": len(rows)}


def _sigma35_ordinal_audit_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    candidates = [dict(value) for value in _require_param(params, "candidates")]
    sig35_order_path = str(_require_param(params, "sig35_order_path"))
    bootstrap_iterations = int(_optional_param(params, "bootstrap_iterations", default=200))
    permutations = int(_optional_param(params, "permutations", default=200))
    seed = int(_optional_param(params, "seed", default=context.config.defaults.random_seed))
    balance_columns = [
        str(value) for value in _optional_param(params, "balance_columns", default=["design_family", "spacer_length"])
    ]
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    order_config = _load_sig35_order(context, relative_path=sig35_order_path)
    inputs.append(ScalarInputRef(kind="workspace_input", artifact_id="sig35_order", path=order_config["path"]))
    ranks = dict(order_config["ranks"])
    for offset, candidate in enumerate(candidates):
        rng = np.random.default_rng(seed + offset)
        candidate_sample = _load_candidate_sample(context, candidate)
        inputs.extend(candidate_sample.inputs)
        normalized = _normalized_geometry_rows(candidate_sample.matrix)
        global_spearman, global_kendall = _sig35_global_statistics(normalized, candidate_sample.rows, ranks=ranks)
        global_groups = group_indices(
            candidate_sample.rows,
            column="sig35_variant",
            exclude_values={"control"},
            allowed_values=set(ranks),
        )
        ci_lower, ci_upper = bootstrap_ci(
            lambda groups=global_groups, rng=rng: _sig35_statistics_from_groups(
                normalized,
                resample_groups(groups, rng=rng),
                ranks=ranks,
            )[0],
            iterations=bootstrap_iterations,
        )
        rows.append(
            _metric_row(
                descriptor=candidate_sample.descriptor,
                metric_id="sig35_ordinal_spearman",
                metric_value=global_spearman,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                extra={
                    "sig35_order_source": order_config["source"],
                    "exploratory": bool(order_config["exploratory"]),
                },
            )
        )
        rows.append(
            _metric_row(
                descriptor=candidate_sample.descriptor,
                metric_id="sig35_ordinal_kendall",
                metric_value=global_kendall,
                extra={
                    "sig35_order_source": order_config["source"],
                    "exploratory": bool(order_config["exploratory"]),
                },
            )
        )

        balanced_groups = balanced_group_indices(
            candidate_sample.rows,
            group_column="sig35_variant",
            balance_columns=balance_columns,
            required_group_values=set(ranks),
            exclude_group_values={"control"},
            rng=rng,
        )
        balanced_spearman = float("nan")
        if balanced_groups:
            centroids = centroid_map(normalized, balanced_groups)
            gaps, distances = ordinal_gap_and_distance_vectors(centroids=centroids, ranks=ranks)
            balanced_spearman = _spearman_correlation(gaps, distances) if gaps.size else float("nan")
        ci_lower, ci_upper = bootstrap_ci(
            lambda groups=balanced_groups, rng=rng: _sig35_statistics_from_groups(
                normalized,
                resample_groups(groups, rng=rng),
                ranks=ranks,
            )[0],
            iterations=bootstrap_iterations,
        )
        rows.append(
            _metric_row(
                descriptor=candidate_sample.descriptor,
                metric_id="sig35_balanced_ordinal_spearman",
                metric_value=balanced_spearman,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                extra={
                    "sig35_order_source": order_config["source"],
                    "exploratory": bool(order_config["exploratory"]),
                },
            )
        )

        within_family_mean = _sig35_mean_statistic_within_groups(
            normalized,
            candidate_sample.rows,
            outer_column="design_family",
            ranks=ranks,
        )
        family_groups = group_indices(candidate_sample.rows, column="design_family", exclude_values={"control"})
        ci_lower, ci_upper = bootstrap_ci(
            lambda groups=family_groups, rng=rng: _sig35_mean_statistic_from_outer_groups(
                normalized,
                candidate_sample.rows,
                outer_groups=resample_groups(groups, rng=rng),
                ranks=ranks,
            ),
            iterations=bootstrap_iterations,
        )
        rows.append(
            _metric_row(
                descriptor=candidate_sample.descriptor,
                metric_id="sig35_within_family_mean_spearman",
                metric_value=within_family_mean,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                extra={
                    "sig35_order_source": order_config["source"],
                    "exploratory": bool(order_config["exploratory"]),
                },
            )
        )

        within_regulator_mean = _sig35_mean_statistic_within_groups(
            normalized,
            candidate_sample.rows,
            outer_column="design_regulator_composition",
            ranks=ranks,
        )
        regulator_groups = group_indices(
            candidate_sample.rows,
            column="design_regulator_composition",
            exclude_values={"control"},
        )
        ci_lower, ci_upper = bootstrap_ci(
            lambda groups=regulator_groups, rng=rng: _sig35_mean_statistic_from_outer_groups(
                normalized,
                candidate_sample.rows,
                outer_groups=resample_groups(groups, rng=rng),
                ranks=ranks,
            ),
            iterations=bootstrap_iterations,
        )
        rows.append(
            _metric_row(
                descriptor=candidate_sample.descriptor,
                metric_id="sig35_within_regulator_mean_spearman",
                metric_value=within_regulator_mean,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                extra={
                    "sig35_order_source": order_config["source"],
                    "exploratory": bool(order_config["exploratory"]),
                },
            )
        )

        observed = global_spearman
        permutation_values: list[float] = []
        variants = [variant for variant in sorted(ranks) if variant != "control"]
        if np.isfinite(observed) and len(variants) >= 3:
            centroids = centroid_map(normalized, global_groups)
            for _ in range(permutations):
                shuffled = rng.permutation([ranks[variant] for variant in variants]).tolist()
                shuffled_ranks = {variant: rank for variant, rank in zip(variants, shuffled, strict=True)}
                gaps, distances = ordinal_gap_and_distance_vectors(centroids=centroids, ranks=shuffled_ranks)
                if gaps.size:
                    permutation_values.append(_spearman_correlation(gaps, distances))
        permutation_pvalue = (
            float(
                (1 + np.sum(np.abs(np.asarray(permutation_values, dtype=np.float64)) >= abs(observed)))
                / (len(permutation_values) + 1)
            )
            if permutation_values and np.isfinite(observed)
            else float("nan")
        )
        rows.append(
            _metric_row(
                descriptor=candidate_sample.descriptor,
                metric_id="sig35_label_permutation_pvalue",
                metric_value=permutation_pvalue,
                extra={
                    "sig35_order_source": order_config["source"],
                    "exploratory": bool(order_config["exploratory"]),
                },
            )
        )
    return pa.Table.from_pylist(rows), inputs, {"candidate_count": len(candidates), "rows": len(rows)}


def _context_robustness_summary_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    pairs = [dict(value) for value in _require_param(params, "pairs")]
    sample_size = int(_optional_param(params, "sample_size", default=4096))
    sample_group_column = _optional_param(params, "sample_group_column", default="design_family")
    seed = int(_optional_param(params, "seed", default=context.config.defaults.random_seed))
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    skipped_metric_ids: list[str] = []
    axes = [
        ("design_family", "design_family_retention_correlation", {"control"}),
        ("design_regulator_composition", "design_regulator_composition_retention_correlation", {"control"}),
        ("sig35_variant", "sig35_variant_retention_correlation", {"control"}),
    ]
    for offset, pair in enumerate(pairs):
        alignment_id = str(_require_param(pair, "alignment_id"))
        left_view_id = str(_require_param(pair, "anchor_view_id"))
        right_view_id = str(_require_param(pair, "context_view_id"))
        pair_id = str(_optional_param(pair, "pair_id", default=f"{left_view_id}_to_{right_view_id}"))
        descriptor = _candidate_descriptor_from_view(
            context,
            view_id=left_view_id,
            candidate_id=pair_id,
            scope_override="anchor_vs_context",
            label_override=_optional_param(pair, "label", default=None),
        )
        left_matrix, _, left_inputs = _load_view_scope_table(
            context,
            view_id=left_view_id,
            alignment_id=alignment_id,
        )
        right_matrix, metadata_rows, right_inputs = _load_view_scope_table(
            context,
            view_id=right_view_id,
            alignment_id=alignment_id,
        )
        inputs.extend(left_inputs)
        inputs.extend(right_inputs)
        if left_matrix.shape != right_matrix.shape:
            raise ContractViolationError("context robustness summary requires aligned anchor/context matrices")
        if sample_size > 0 and sample_size < len(metadata_rows):
            sampled_indices = _sample_metadata_indices(
                metadata_rows,
                sample_size=sample_size,
                group_column=sample_group_column,
                seed=seed + offset,
            )
            index_array = np.asarray(sampled_indices, dtype=np.int64)
            left_matrix = left_matrix[index_array]
            right_matrix = right_matrix[index_array]
            metadata_rows = [metadata_rows[index] for index in sampled_indices]
        left_norm = _normalized_geometry_rows(left_matrix)
        right_norm = _normalized_geometry_rows(right_matrix)
        self_cosine = np.asarray(np.sum(left_norm * right_norm, axis=1), dtype=np.float64)
        rows.append(
            _metric_row(
                descriptor=descriptor,
                metric_id="context_self_cosine_median",
                metric_value=float(np.median(self_cosine)),
            )
        )
        for column, metric_id, exclude_values in axes:
            anchor_vector, context_vector = aligned_cohort_distance_vectors(
                left_norm,
                right_norm,
                metadata_rows,
                column=column,
                exclude_values=exclude_values,
            )
            if anchor_vector.size == 0 or context_vector.size == 0:
                skipped_metric_ids.append(metric_id)
                continue
            retention = _pearson_correlation(anchor_vector, context_vector)
            rows.append(_metric_row(descriptor=descriptor, metric_id=metric_id, metric_value=retention))
    return (
        pa.Table.from_pylist(rows),
        inputs,
        {
            "pair_count": len(pairs),
            "rows": len(rows),
            "skipped_metric_ids": skipped_metric_ids,
        },
    )


def _reference_alignment_summary_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    candidates = [dict(value) for value in _require_param(params, "candidates")]
    reference_group_columns = [
        str(value)
        for value in _optional_param(
            params,
            "reference_group_columns",
            default=[],
        )
    ]
    reference_label_column = str(_optional_param(params, "reference_label_column", default="usr_label__primary"))
    min_reference_group_size = int(_optional_param(params, "min_reference_group_size", default=2))
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    for candidate in candidates:
        candidate_sample = _load_candidate_sample(context, candidate)
        inputs.extend(candidate_sample.inputs)
        normalized = _normalized_geometry_rows(candidate_sample.matrix)
        design_groups = group_indices(candidate_sample.rows, column="design_family")
        reference_groups = group_indices(candidate_sample.rows, column="usr_label__primary")
        emitted_rows = 0
        if (
            {"background_only", "ethanol", "ciprofloxacin"}.issubset(design_groups)
            and any(label.lower() == "spyp" for label in reference_groups)
            and any(label.lower() == "sulap" for label in reference_groups)
        ):
            centroids = centroid_map(normalized, design_groups)
            reference_centroids = {
                label.lower(): centroid
                for label, indices in reference_groups.items()
                if (centroid := try_l2_normalize_vector(np.asarray(normalized[indices].mean(axis=0), dtype=np.float32)))
                is not None
            }
            ethanol_alignment = float(np.dot(centroids["ethanol"], reference_centroids["spyp"])) - float(
                np.dot(centroids["background_only"], reference_centroids["spyp"])
            )
            cipro_alignment = float(np.dot(centroids["ciprofloxacin"], reference_centroids["sulap"])) - float(
                np.dot(centroids["background_only"], reference_centroids["sulap"])
            )
            rows.extend(
                [
                    _metric_row(
                        descriptor=candidate_sample.descriptor,
                        metric_id="reference_alignment_ethanol_background_relative",
                        metric_value=ethanol_alignment,
                    ),
                    _metric_row(
                        descriptor=candidate_sample.descriptor,
                        metric_id="reference_alignment_cipro_background_relative",
                        metric_value=cipro_alignment,
                    ),
                ]
            )
            emitted_rows += 2
        elif not reference_group_columns:
            if not {"background_only", "ethanol", "ciprofloxacin"}.issubset(design_groups):
                raise ContractViolationError(
                    "reference_alignment_summary requires background_only, ethanol, "
                    f"and ciprofloxacin cohorts in {_require_param(candidate, 'view_id')!r}"
                )
            raise ContractViolationError(
                "reference_alignment_summary requires carried SpyP and SulA rows in "
                f"{_require_param(candidate, 'view_id')!r}"
            )
        reference_indices = [
            index
            for index, row in enumerate(candidate_sample.rows)
            if row.get(reference_label_column) is not None and str(row.get(reference_label_column)).strip()
        ]
        for group_column in reference_group_columns:
            grouped: dict[str, list[int]] = {}
            for index in reference_indices:
                value = candidate_sample.rows[index].get(group_column)
                if value is None or not str(value).strip() or str(value).lower() == "nan":
                    continue
                grouped.setdefault(str(value), []).append(index)
            for group_value, indices in sorted(grouped.items()):
                if len(indices) < min_reference_group_size:
                    continue
                distances = _cosine_distance_upper(np.asarray(normalized[indices], dtype=np.float32))
                distance_median = float(np.median(distances)) if distances.size else 0.0
                distance_iqr = (
                    float(np.percentile(distances, 75.0) - np.percentile(distances, 25.0)) if distances.size else 0.0
                )
                extra = {
                    "reference_group_column": group_column,
                    "reference_group": group_value,
                    "reference_rows": len(indices),
                    "category": f"{group_column}: {group_value}",
                    "label": group_value,
                }
                rows.extend(
                    [
                        _metric_row(
                            descriptor=candidate_sample.descriptor,
                            metric_id="reference_group_size",
                            metric_value=float(len(indices)),
                            category="reference collapse",
                            extra={
                                **extra,
                                "display_name": _reference_group_panel_title(
                                    metric_id="reference_group_size",
                                    group_column=group_column,
                                    group_value=group_value,
                                ),
                            },
                        ),
                        _metric_row(
                            descriptor=candidate_sample.descriptor,
                            metric_id="reference_group_pairwise_cosine_distance_median",
                            metric_value=distance_median,
                            category="reference collapse",
                            extra={
                                **extra,
                                "display_name": _reference_group_panel_title(
                                    metric_id="reference_group_pairwise_cosine_distance_median",
                                    group_column=group_column,
                                    group_value=group_value,
                                ),
                            },
                        ),
                        _metric_row(
                            descriptor=candidate_sample.descriptor,
                            metric_id="reference_group_pairwise_cosine_distance_iqr",
                            metric_value=distance_iqr,
                            category="reference collapse",
                            extra={
                                **extra,
                                "display_name": _reference_group_panel_title(
                                    metric_id="reference_group_pairwise_cosine_distance_iqr",
                                    group_column=group_column,
                                    group_value=group_value,
                                ),
                            },
                        ),
                    ]
                )
                emitted_rows += 3
        if emitted_rows == 0:
            raise ContractViolationError(
                "reference_alignment_summary requires SpyP/SulA alignment rows or at least one "
                f"reference group with >= {min_reference_group_size} rows in {_require_param(candidate, 'view_id')!r}"
            )
    for row in rows:
        row.setdefault("reference_group_column", None)
        row.setdefault("reference_group", None)
        row.setdefault("reference_rows", None)
    return pa.Table.from_pylist(rows), inputs, {"candidate_count": len(candidates), "rows": len(rows)}


def _context_pair_summary_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    comparisons = [dict(value) for value in _require_param(params, "comparisons")]
    metric_specs = [
        ("context_self_cosine", "context_self_cosine_median"),
        ("context_shift_l2", "context_shift_l2_median"),
    ]
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    for comparison in comparisons:
        scalar_id = str(_require_param(comparison, "scalar_id"))
        comparison_id = str(_require_param(comparison, "comparison_id"))
        comparison_label = str(_require_param(comparison, "comparison_label"))
        comparison_role = str(_require_param(comparison, "comparison_role"))
        source_rows, source_inputs = _load_scalar_rows(context, scalar_id=scalar_id)
        inputs.extend(source_inputs)
        for source_column, metric_id in metric_specs:
            values = [
                float(value)
                for row in source_rows
                if (value := row.get(source_column)) is not None and np.isfinite(float(value))
            ]
            metric_value = float(np.median(np.asarray(values, dtype=np.float64))) if values else float("nan")
            rows.append(
                _metric_row(
                    descriptor={
                        "comparison_id": comparison_id,
                        "comparison_label": comparison_label,
                        "comparison_role": comparison_role,
                    },
                    metric_id=metric_id,
                    metric_value=metric_value,
                    extra={
                        "label": comparison_label,
                        "source_scalar_id": scalar_id,
                    },
                )
            )
    return pa.Table.from_pylist(rows), inputs, {"comparison_count": len(comparisons), "rows": len(rows)}


def _candidate_decision_frontier_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    health_scalar = str(_require_param(params, "health_scalar"))
    design_scalar = str(_require_param(params, "design_scalar"))
    sigma35_scalar = str(_require_param(params, "sigma35_scalar"))
    context_scalar = str(_require_param(params, "context_scalar"))
    health_metric_id = str(_optional_param(params, "health_metric_id", default="effective_rank"))
    design_metric_id = str(
        _optional_param(params, "design_metric_id", default="design_family_balanced_separation_ratio")
    )
    sigma35_metric_id = str(_optional_param(params, "sigma35_metric_id", default="sig35_ordinal_spearman"))
    context_metric_id = str(_optional_param(params, "context_metric_id", default="context_self_cosine_median"))
    candidate_ids = [str(value) for value in _optional_param(params, "candidate_ids", default=[])]
    context_pairs = {
        str(_require_param(entry, "candidate_id")): str(_require_param(entry, "pair_id"))
        for entry in _optional_param(params, "context_pairs", default=[])
    }
    candidate_roles = {
        str(_require_param(entry, "candidate_id")): str(_require_param(entry, "role"))
        for entry in _optional_param(params, "candidate_roles", default=[])
    }
    annotation_labels = {
        str(_require_param(entry, "candidate_id")): str(_require_param(entry, "label"))
        for entry in _optional_param(params, "annotation_labels", default=[])
    }

    health_rows, health_inputs = _load_scalar_rows(context, scalar_id=health_scalar)
    design_rows, design_inputs = _load_scalar_rows(context, scalar_id=design_scalar)
    sigma35_rows, sigma35_inputs = _load_scalar_rows(context, scalar_id=sigma35_scalar)
    context_rows, context_inputs = _load_scalar_rows(context, scalar_id=context_scalar)
    inputs = [*health_inputs, *design_inputs, *sigma35_inputs, *context_inputs]

    health_map = _rows_by_candidate_and_metric(health_rows)
    design_map = _rows_by_candidate_and_metric(design_rows)
    sigma35_map = _rows_by_candidate_and_metric(sigma35_rows)
    context_map = _rows_by_candidate_and_metric(context_rows)

    ordered_candidate_ids = candidate_ids or list(health_map)
    rows: list[dict[str, object]] = []
    for index, candidate_id in enumerate(ordered_candidate_ids):
        health_metrics = health_map.get(candidate_id, {})
        descriptor_source = (
            health_metrics.get(health_metric_id)
            or next(iter(health_metrics.values()), None)
            or next(iter(design_map.get(candidate_id, {}).values()), None)
            or next(iter(sigma35_map.get(candidate_id, {}).values()), None)
        )
        if descriptor_source is None:
            raise ContractViolationError(f"candidate_decision_frontier is missing descriptor rows for {candidate_id!r}")
        context_pair_id = context_pairs.get(candidate_id)
        context_metric_row = (
            context_map.get(context_pair_id, {}).get(context_metric_id) if context_pair_id is not None else None
        )
        health_metric_row = health_metrics.get(health_metric_id)
        design_metric_row = design_map.get(candidate_id, {}).get(design_metric_id)
        sigma35_metric_row = sigma35_map.get(candidate_id, {}).get(sigma35_metric_id)
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_label": descriptor_source["candidate_label"],
                "candidate_family": descriptor_source["candidate_family"],
                "candidate_model": descriptor_source["candidate_model"],
                "candidate_scope": descriptor_source["candidate_scope"],
                "candidate_order": index,
                "selection_role": candidate_roles.get(candidate_id, "candidate"),
                "frontier_label": annotation_labels.get(candidate_id, str(descriptor_source["candidate_label"])),
                "health_status": str(descriptor_source.get("health_status") or "unknown"),
                "collapse_flag": bool(descriptor_source.get("collapse_flag", False)),
                "effective_rank": (
                    float(health_metric_row["metric_value"]) if health_metric_row is not None else float("nan")
                ),
                "design_family_balanced_separation_ratio": (
                    float(design_metric_row["metric_value"]) if design_metric_row is not None else float("nan")
                ),
                "sig35_ordinal_spearman": (
                    float(sigma35_metric_row["metric_value"]) if sigma35_metric_row is not None else float("nan")
                ),
                "context_self_cosine_median": (
                    float(context_metric_row["metric_value"]) if context_metric_row is not None else float("nan")
                ),
                "context_pair_id": context_pair_id,
                "context_validation_status": "direct" if context_pair_id is not None else "not_applicable",
                "x_display_name": "Balanced design-family separation ratio",
                "y_display_name": "Sigma-35 ordinal Spearman",
            }
        )
    return pa.Table.from_pylist(rows), inputs, {"candidate_count": len(rows), "rows": len(rows)}


def _sigma35_centroid_distance_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    candidates = [dict(value) for value in _require_param(params, "candidates")]
    sig35_order_path = str(_require_param(params, "sig35_order_path"))
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    order_config = _load_sig35_order(context, relative_path=sig35_order_path)
    inputs.append(ScalarInputRef(kind="workspace_input", artifact_id="sig35_order", path=order_config["path"]))
    ranks = dict(order_config["ranks"])
    sequences = dict(order_config["sequences"])
    for candidate in candidates:
        candidate_sample = _load_candidate_sample(context, candidate)
        inputs.extend(candidate_sample.inputs)
        normalized = _normalized_geometry_rows(candidate_sample.matrix)
        groups = group_indices(
            candidate_sample.rows,
            column="sig35_variant",
            exclude_values={"control"},
        )
        unranked_variants = sorted(set(groups) - set(ranks), key=str.casefold)
        ordered_variants = [
            variant for variant, _ in sorted(ranks.items(), key=lambda item: int(item[1])) if variant in groups
        ]
        ordered_variants.extend(unranked_variants)
        variant_labels = {
            variant: f"{sequences[variant]} ({variant})"
            for variant in ordered_variants
            if variant in sequences and sequences[variant]
        }
        variant_labels.update(
            {
                variant: f"{variant} (annotated, unranked)" if variant in unranked_variants else f"variant {variant}"
                for variant in ordered_variants
                if variant not in variant_labels
            }
        )
        centroids = centroid_map(normalized, groups)
        for row_variant in ordered_variants:
            for column_variant in ordered_variants:
                value = float("nan")
                if row_variant in centroids and column_variant in centroids:
                    value = 1.0 - float(np.dot(centroids[row_variant], centroids[column_variant]))
                rows.append(
                    {
                        **candidate_sample.descriptor,
                        "row_variant": variant_labels[row_variant],
                        "column_variant": variant_labels[column_variant],
                        "metric_value": value,
                    }
                )
    return pa.Table.from_pylist(rows), inputs, {"candidate_count": len(candidates), "rows": len(rows)}


def _sample_metadata_indices(
    rows: list[dict[str, object]],
    *,
    sample_size: int,
    group_column: str | None,
    seed: int,
) -> list[int]:
    row_count = len(rows)
    if sample_size >= row_count:
        return list(range(row_count))
    rng = np.random.default_rng(seed)
    if group_column is None:
        return sorted(rng.choice(row_count, size=sample_size, replace=False).tolist())
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        key = str(row.get(group_column))
        groups.setdefault(key, []).append(index)
    total_rows = sum(len(indices) for indices in groups.values())
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for key, indices in groups.items():
        raw = (len(indices) / total_rows) * sample_size
        count = min(len(indices), int(raw))
        quotas[key] = count
        assigned += count
        remainders.append((raw - count, key))
    for _, key in sorted(remainders, reverse=True):
        if assigned >= sample_size:
            break
        if quotas[key] >= len(groups[key]):
            continue
        quotas[key] += 1
        assigned += 1
    selected: list[int] = []
    for key in sorted(groups):
        candidates = np.asarray(groups[key], dtype=np.int64)
        order = rng.permutation(len(candidates))
        selected.extend(sorted(candidates[order][: quotas[key]].tolist()))
    return sorted(selected)


_PREASSAY_BUILDERS: dict[str, ScalarTableBuilder] = {
    "representation_health_summary": _representation_health_summary_table,
    "design_structure_summary": _design_structure_summary_table,
    "sigma35_ordinal_audit": _sigma35_ordinal_audit_table,
    "context_robustness_summary": _context_robustness_summary_table,
    "context_pair_summary": _context_pair_summary_table,
    "reference_alignment_summary": _reference_alignment_summary_table,
    "candidate_decision_frontier": _candidate_decision_frontier_table,
    "sigma35_centroid_distance": _sigma35_centroid_distance_table,
}

PREASSAY_BUILDER_KINDS = frozenset(_PREASSAY_BUILDERS)


def build_preassay_scalar_artifact(
    context: WorkspaceContext,
    *,
    scalar_id: str,
    builder_kind: str,
    params: dict[str, Any],
) -> BuiltScalarArtifact | None:
    builder = _PREASSAY_BUILDERS.get(builder_kind)
    if builder is None:
        return None
    table, inputs, stats = builder(context, params)
    artifact_dir = context.output_root / "scalars" / scalar_id
    write_table(table, artifact_dir / "table.parquet")
    return BuiltScalarArtifact(
        artifact_dir=artifact_dir,
        rows=table.num_rows,
        columns=table.column_names,
        inputs=inputs,
        outputs=[],
        stats=stats,
    )
