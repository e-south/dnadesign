"""Metrics loading, enrichment, and review status helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..constants import NULL_ORACLE_ID, ORACLE_ID
from ..decision import (
    _decision_from_metrics,
    decision_reasons_from_metrics,
    enrich_metric_rows,
    gate_results_from_metrics,
    metric_definitions,
    metric_quality_from_metrics,
)


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"metrics.json not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"metrics.json is malformed: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("metrics.json must contain a JSON object")
    if not isinstance(payload.get("safety"), dict):
        raise RuntimeError("metrics.json missing object field: safety")
    if not isinstance(payload.get("runs"), list):
        raise RuntimeError("metrics.json missing list field: runs")
    return payload


def _enriched_metrics_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    runs = payload.get("runs") or []
    out["runs"] = enrich_metric_rows([row for row in runs if isinstance(row, Mapping)])
    rounds = payload.get("rounds") or []
    out["rounds"] = enrich_metric_rows([row for row in rounds if isinstance(row, Mapping)])
    safety = payload.get("safety") if isinstance(payload.get("safety"), Mapping) else {}
    decision = payload.get("decision")
    if isinstance(decision, str) and decision:
        out["decision"] = decision
    out["gate_results"] = gate_results_from_metrics(out["runs"], safety)
    out["decision_reasons"] = decision_reasons_from_metrics(
        out["runs"],
        safety,
        decision=str(decision) if decision else None,
    )
    out["metric_quality"] = metric_quality_from_metrics(out["runs"])
    out["metric_definitions"] = metric_definitions()
    return out


def _review_decision(metrics_payload: Mapping[str, Any]) -> str:
    safety = metrics_payload.get("safety")
    runs = metrics_payload.get("runs")
    if not isinstance(safety, Mapping) or not isinstance(runs, list):
        raise RuntimeError("metrics.json is missing safety/runs contract fields")
    return _decision_from_metrics([dict(row) for row in runs if isinstance(row, Mapping)], safety)


def _review_problems(*, audit, review_decision: str | None) -> list[str]:
    problems = list(audit.problems)
    if audit.decision and review_decision and audit.decision != review_decision:
        problems.append(f"persisted_decision_mismatch:{audit.decision}!={review_decision}")
    return problems


def _gate_coverage(runs: list[dict[str, Any]]) -> dict[str, Any]:
    campaigns = sorted({str(row.get("campaign")) for row in runs if row.get("campaign")})
    splits = sorted({str(row.get("split_id")) for row in runs if row.get("split_id")})
    pair_counts: dict[tuple[str, str], set[str]] = {}
    for row in runs:
        key = (str(row.get("campaign")), str(row.get("split_id")))
        pair_counts.setdefault(key, set()).add(str(row.get("oracle_id")))
    positive_null_pairs_complete = all({ORACLE_ID, NULL_ORACLE_ID}.issubset(values) for values in pair_counts.values())
    omitted: list[str] = []
    if "ethanol" not in campaigns:
        omitted.append("ethanol")
    if "dual" not in campaigns:
        omitted.append("dual")
    if "leave_sigma35_variant" not in splits:
        omitted.append("leave_sigma35_variant")
    return {
        "campaigns": campaigns,
        "splits": splits,
        "positive_null_pairs_complete": bool(positive_null_pairs_complete) if runs else False,
        "omitted_scored_gates": omitted,
    }
