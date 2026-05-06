"""Shared scalar-builder utilities for artifact-first latentdna workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..geometry.preprocessing import standardize_and_l2_normalize
from ..labels import humanize_candidate
from ..metrics.definitions import resolve_metric_definition
from ..views.scopes import resolve_view_scope
from ..workspaces.loader import WorkspaceContext


@dataclass(frozen=True, slots=True)
class ScalarInputRef:
    kind: str
    artifact_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class BuiltScalarArtifact:
    artifact_dir: Path
    rows: int
    columns: list[str]
    inputs: list[ScalarInputRef]
    outputs: list[tuple[str, str]]
    stats: dict[str, object]


@dataclass(frozen=True, slots=True)
class PairwiseDistanceSummary:
    median: float
    iqr: float
    source_rows: int
    evaluated_rows: int
    pair_count: int
    max_rows: int
    seed: int
    method: str


def _require_param(params: dict[str, Any], key: str) -> Any:
    if key not in params:
        raise ContractViolationError(f"scalar.build requires param {key!r}")
    return params[key]


def _optional_param(params: dict[str, Any], key: str, *, default: Any = None) -> Any:
    return params.get(key, default)


def _reducer_summary_path(context: WorkspaceContext, reducer_id: str) -> Path:
    path = context.output_root / "reducers" / reducer_id / "summary.json"
    if not path.is_file():
        raise MissingArtifactError(f"reducer artifact is missing for scalar.build: {reducer_id}")
    return path


def _workspace_input_path(context: WorkspaceContext, relative_path: str) -> Path:
    path = (context.workspace_dir / relative_path).resolve()
    if not path.is_file():
        raise MissingArtifactError(f"workspace input is missing for scalar.build: {relative_path}")
    return path


def _normalized_geometry_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize study geometry with an explicit collapse-tolerant policy."""

    return standardize_and_l2_normalize(
        matrix,
        nonfinite_policy="error",
        zero_variance_policy="drop_or_zero",
        zero_row_policy="zero",
    )


def _effective_rank(ratios: list[float]) -> float:
    array = np.asarray([float(value) for value in ratios if float(value) > 0.0], dtype=np.float64)
    if array.size == 0:
        return 0.0
    probabilities = array / np.sum(array, dtype=np.float64)
    entropy = -np.sum(probabilities * np.log(probabilities), dtype=np.float64)
    return float(np.exp(entropy))


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = np.asarray(values[order], dtype=np.float64)
    start = 0
    while start < values.size:
        end = start
        while end + 1 < values.size and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        rank_value = (start + end + 2) / 2.0
        ranks[order[start : end + 1]] = rank_value
        start = end + 1
    return ranks


def _pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    if x.size == 1:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    if x.size == 1:
        return float("nan")
    return _pearson_correlation(_rankdata_average(x), _rankdata_average(y))


def _kendall_tau(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")
    concordant = 0
    discordant = 0
    tie_x = 0
    tie_y = 0
    for i in range(x.size):
        for j in range(i + 1, x.size):
            dx = np.sign(x[i] - x[j])
            dy = np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_x += 1
                continue
            if dy == 0:
                tie_y += 1
                continue
            if dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = np.sqrt(float((concordant + discordant + tie_x) * (concordant + discordant + tie_y)))
    if denominator < 1e-12:
        return float("nan")
    return float((concordant - discordant) / denominator)


def _pairwise_cosine_distance_summary(
    matrix: np.ndarray,
    *,
    max_rows: int = 4096,
    seed: int = 17,
) -> PairwiseDistanceSummary:
    if max_rows < 2:
        raise ContractViolationError("pairwise cosine distance summaries require max_rows >= 2")
    normalized = _normalized_geometry_rows(matrix)
    source_rows = int(normalized.shape[0])
    if source_rows > max_rows:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(source_rows, size=max_rows, replace=False))
        normalized = np.ascontiguousarray(normalized[indices])
        method = "seeded_row_sample_all_pairs"
    else:
        method = "exact_all_pairs"
    distances = _cosine_distance_upper_from_normalized(normalized)
    median = float(np.median(distances)) if distances.size else float("nan")
    iqr = float(np.percentile(distances, 75.0) - np.percentile(distances, 25.0)) if distances.size else float("nan")
    return PairwiseDistanceSummary(
        median=median,
        iqr=iqr,
        source_rows=source_rows,
        evaluated_rows=int(normalized.shape[0]),
        pair_count=int(distances.size),
        max_rows=max_rows,
        seed=seed,
        method=method,
    )


def _cosine_distance_upper_from_normalized(normalized: np.ndarray) -> np.ndarray:
    distances = 1.0 - np.asarray(normalized @ normalized.T, dtype=np.float32)
    distances = np.clip(distances, 0.0, 2.0)
    upper = np.triu_indices(distances.shape[0], k=1)
    return np.asarray(distances[upper], dtype=np.float64)


def _candidate_descriptor_from_view(
    context: WorkspaceContext,
    *,
    view_id: str,
    candidate_id: str | None = None,
    scope_override: str | None = None,
    label_override: str | None = None,
) -> dict[str, object]:
    view = context.require_view(view_id)
    tags = dict(getattr(view, "tags", {}) or {})
    family = str(tags.get("family") or "")
    model = str(tags.get("model") or "").lower()
    scope = str(scope_override or tags.get("scope") or "")
    label = label_override or humanize_candidate(
        {
            "candidate_model": f"evo2_{model}" if model else "",
            "candidate_scope": scope,
            "candidate_family": family,
        }
    )
    return {
        "candidate_id": candidate_id or view_id,
        "candidate_family": family,
        "candidate_model": model,
        "candidate_scope": scope,
        "candidate_label": label,
    }


def _metric_row(
    *,
    context: WorkspaceContext | None = None,
    descriptor: dict[str, object],
    metric_id: str,
    metric_value: float,
    category: str | None = None,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    definition = resolve_metric_definition(metric_id, config=context.config if context is not None else None)
    row = {
        **descriptor,
        "metric_id": definition.metric_id,
        "metric_name": definition.metric_id,
        "metric_value": float(metric_value),
        "value": float(metric_value),
        "metric_family": definition.metric_family,
        "evidence_tier": definition.evidence_tier,
        "task_id": definition.task_id,
        "mathematical_definition": definition.mathematical_definition,
        "unit": definition.unit,
        "direction": definition.direction,
        "aggregation_level": definition.aggregation_level,
        "higher_is_better": definition.higher_is_better,
        "display_name": definition.display_name,
        "definition_version": definition.definition_version,
        "category": category or definition.metric_id,
        "label": definition.metric_id,
    }
    if ci_lower is not None:
        row["ci_lower"] = float(ci_lower)
    if ci_upper is not None:
        row["ci_upper"] = float(ci_upper)
    if extra:
        row.update(extra)
    return row


def _load_view_scope_table(
    context: WorkspaceContext,
    *,
    view_id: str,
    sample_id: str | None = None,
    alignment_id: str | None = None,
) -> tuple[np.ndarray, list[dict[str, object]], list[ScalarInputRef]]:
    matrix, rows_table, input_kind, input_id = resolve_view_scope(
        context,
        view_id=view_id,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )
    if input_kind == "sample_set":
        artifact_id = sample_id
        artifact_path = context.output_root / "samples" / str(sample_id) / "rows.parquet"
    elif input_kind == "alignment_set":
        artifact_id = alignment_id
        artifact_path = context.output_root / "alignments" / str(alignment_id) / "rows.parquet"
    else:
        artifact_id = input_id
        artifact_path = context.output_root / "views" / view_id / "rows.parquet"
    inputs = [
        ScalarInputRef(
            kind="view_matrix",
            artifact_id=view_id,
            path=context.output_root / "views" / view_id / "matrix.npy",
        ),
        ScalarInputRef(kind=input_kind, artifact_id=str(artifact_id), path=artifact_path),
    ]
    return np.asarray(matrix, dtype=np.float32), rows_table.to_pylist(), inputs
