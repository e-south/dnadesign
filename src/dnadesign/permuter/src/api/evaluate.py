"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/api/evaluate.py

Filesystem-free public scoring API.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from dnadesign.permuter.src.api.contracts import (
    EvaluatorPlan,
    Metadata,
    MetricSpec,
    PermuterResult,
    VariantRecord,
)
from dnadesign.permuter.src.contracts.metrics import observed_metric_column
from dnadesign.permuter.src.core.registry import get_evaluator
from dnadesign.permuter.src.evaluators.results import normalize_scores


def evaluate_variants(result: PermuterResult, plan: EvaluatorPlan) -> PermuterResult:
    """Score in-memory variants with registered evaluators and return a new result."""

    metrics = tuple(plan.metrics)
    _assert_metric_specs(metrics)
    records = tuple(result.records)
    sequences = [record.sequence for record in records]
    out_records = list(records)
    ref_sequence = plan.ref_sequence if plan.ref_sequence is not None else result.reference_sequence

    for spec in metrics:
        ev_cls = get_evaluator(spec.evaluator)
        ev = ev_cls(**dict(spec.params or {}))
        scores = ev.score(
            sequences,
            metric=spec.metric,
            ref_sequence=ref_sequence,
            ref_embedding=None,
        )
        columns = normalize_scores(scores, n=len(records), metric_id=spec.id)
        out_records = [
            _record_with_observed_metric(
                record,
                metric_id=spec.id,
                columns=columns,
                row_index=i,
                overwrite=plan.overwrite,
            )
            for i, record in enumerate(out_records)
        ]

    return replace(
        result,
        records=tuple(out_records),
        metadata=_result_metadata_with_evaluations(result.metadata, metrics),
    )


def _assert_metric_specs(metrics: tuple[MetricSpec, ...]) -> None:
    if not metrics:
        raise ValueError("EvaluatorPlan.metrics must contain at least one MetricSpec")
    ids = [str(spec.id).strip() for spec in metrics]
    if any(not metric_id for metric_id in ids):
        raise ValueError("MetricSpec.id is required")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate MetricSpec.id values are not allowed: {ids}")
    for spec in metrics:
        if not str(spec.evaluator or "").strip():
            raise ValueError(f"MetricSpec.evaluator is required for metric {spec.id!r}")
        if not str(spec.metric or "").strip():
            raise ValueError(f"MetricSpec.metric is required for metric {spec.id!r}")


def _record_with_observed_metric(
    record: VariantRecord,
    *,
    metric_id: str,
    columns,
    row_index: int,
    overwrite: bool,
) -> VariantRecord:
    metadata = dict(record.metadata)
    permuter = _permuter_payload(metadata, record_id=record.id)
    observed = _observed_payload(permuter, record_id=record.id)
    if metric_id in observed and not overwrite:
        raise ValueError(f"{record.id}: observed metric {metric_id!r} already exists; set overwrite=True to replace it")
    observed[metric_id] = _metric_payload(metric_id=metric_id, columns=columns, row_index=row_index)
    permuter["observed"] = observed
    metadata["permuter"] = permuter
    return replace(record, metadata=metadata)


def _metric_payload(*, metric_id: str, columns, row_index: int) -> object:
    base = observed_metric_column(metric_id)
    if set(columns) == {base}:
        return _jsonish(columns[base].iloc[row_index])
    values: dict[str, object] = {}
    prefix = f"{base}__"
    for column, series in sorted(columns.items()):
        if not str(column).startswith(prefix):
            raise ValueError(f"Observed subcolumn for {metric_id!r} has unexpected name: {column!r}")
        suffix = str(column).removeprefix(prefix)
        values[suffix] = _jsonish(series.iloc[row_index])
    return values


def _permuter_payload(metadata: dict[str, object], *, record_id: str) -> dict[str, object]:
    payload = metadata.get("permuter")
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{record_id}: metadata.permuter must be a mapping")
    return dict(payload)


def _observed_payload(permuter: dict[str, object], *, record_id: str) -> dict[str, object]:
    payload = permuter.get("observed")
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{record_id}: metadata.permuter.observed must be a mapping")
    return dict(payload)


def _result_metadata_with_evaluations(metadata: Metadata, metrics: tuple[MetricSpec, ...]) -> dict[str, object]:
    out = dict(metadata)
    permuter = out.get("permuter")
    if permuter is None:
        payload: dict[str, object] = {}
    elif isinstance(permuter, Mapping):
        payload = dict(permuter)
    else:
        raise ValueError("PermuterResult.metadata.permuter must be a mapping")
    prior = payload.get("evaluations", ())
    if prior is None:
        evaluations: list[object] = []
    elif isinstance(prior, (list, tuple)):
        evaluations = list(prior)
    else:
        raise ValueError("PermuterResult.metadata.permuter.evaluations must be a list")
    evaluations.extend(
        {
            "id": spec.id,
            "evaluator": spec.evaluator,
            "metric": spec.metric,
            "params": dict(spec.params or {}),
        }
        for spec in metrics
    )
    payload["evaluations"] = evaluations
    out["permuter"] = payload
    return out


def _jsonish(value: object) -> object:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    if isinstance(value, tuple):
        return [_jsonish(v) for v in value]
    if isinstance(value, list):
        return [_jsonish(v) for v in value]
    return value
