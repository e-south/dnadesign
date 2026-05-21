"""Study-owned DenseGen axis OPAL probe package."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import NULL_ORACLE_ID, ORACLE_ID

PASS_CIPRO_RANDOM_GATE = "PASS_CIPRO_RANDOM_GATE"
PASS_RANDOM_ALL_GATE = "PASS_RANDOM_ALL_GATE"
PASS_LEAVE_SIGMA35_GATE = "PASS_LEAVE_SIGMA35_GATE"
PASS_FULL_MATRIX_GATE = "PASS_FULL_MATRIX_GATE"
PASS_SCOPED_GATE = "PASS_SCOPED_GATE"
NULL_LIFT_STOP_THRESHOLD = 1.25


def metric_definitions() -> dict[str, str]:
    return {
        "selected_target_count": "How many of OPAL's selected K candidates match the run target class.",
        "precision_at_k": "selected_target_count divided by K.",
        "target_prevalence": "Fraction of evaluable candidates in the current split pool that match the target class.",
        "lift": "lift = precision@K / target prevalence; values above 1 mean enrichment over random selection.",
        "binomial_tail_p": (
            "Approximate probability of seeing at least the observed selected_target_count under random selection "
            "from a pool with the same target prevalence."
        ),
        "null_lift": (
            "The same lift calculation for a campaign trained on permuted labels, still evaluated against the true "
            "target class. High null lift means the null run enriched true targets and the positive signal is not "
            "clean."
        ),
        "round": (
            "Round metrics are computed retroactively from each OPAL prediction ledger. Round 0 uses the initial "
            "labels; later rounds include the previous round's selected labels."
        ),
    }


def _finite_metric(row: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key, np.nan))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _binomial_tail_ge(successes: int, trials: int, probability: float | None) -> float | None:
    if trials <= 0 or successes < 0 or successes > trials:
        return None
    try:
        p = float(probability)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        return None
    return float(
        sum(
            math.comb(trials, index) * (p**index) * ((1.0 - p) ** (trials - index))
            for index in range(successes, trials + 1)
        )
    )


def _decision_from_metrics(metrics: list[dict[str, Any]], safety: Mapping[str, Any]) -> str:
    if not safety.get("path_safety_pass") or not safety.get("forbidden_input_pass") or not safety.get("x_surface_pass"):
        return "STOP"
    if not metrics:
        return "PENDING"
    if any(row.get("status") == "missing_predictions" for row in metrics):
        raise ValueError("metrics contain missing_predictions; scored runs must fail before decision")
    evaluable = [row for row in metrics if row.get("status") != "missing_predictions"]
    if not evaluable:
        return "DEBUG"

    null_lifts = [
        value
        for row in evaluable
        if row.get("oracle_id") == NULL_ORACLE_ID
        for value in [_finite_metric(row, "target_lift_at_k_true")]
        if value is not None
    ]
    if null_lifts and max(null_lifts) > NULL_LIFT_STOP_THRESHOLD:
        return "STOP"
    pair_keys = sorted({(str(row.get("campaign")), str(row.get("split_id"))) for row in evaluable})
    if not pair_keys:
        return "DEBUG"
    for campaign, split_id in pair_keys:
        positive = [
            value
            for row in evaluable
            if row.get("campaign") == campaign and row.get("split_id") == split_id and row.get("oracle_id") == ORACLE_ID
            for value in [_finite_metric(row, "target_lift_at_k_true")]
            if value is not None
        ]
        null = [
            value
            for row in evaluable
            if row.get("campaign") == campaign
            and row.get("split_id") == split_id
            and row.get("oracle_id") == NULL_ORACLE_ID
            for value in [_finite_metric(row, "target_lift_at_k_true")]
            if value is not None
        ]
        if not positive or not null:
            return "DEBUG"
        if max(positive) <= max(null):
            return "DEBUG"
    return _pass_decision_for_coverage(evaluable)


def enrich_metric_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return run metrics with count-aware derived fields for reports."""
    enriched: list[dict[str, Any]] = []
    for row in metrics:
        out = dict(row)
        selection_k = _int_or_none(out.get("selection_k"))
        precision_true = _finite_metric(out, "selected_target_precision_at_k_true")
        prevalence_true = _finite_metric(out, "target_class_prevalence_true")
        precision_oracle = _finite_metric(out, "selected_target_precision_at_k_oracle")
        prevalence_oracle = _finite_metric(out, "target_class_prevalence_oracle")
        if selection_k is not None and selection_k > 0:
            if "selected_target_count_true" not in out and precision_true is not None:
                out["selected_target_count_true"] = int(round(precision_true * selection_k))
            if "selected_target_count_oracle" not in out and precision_oracle is not None:
                out["selected_target_count_oracle"] = int(round(precision_oracle * selection_k))
            if out.get("selected_target_count_true") is not None:
                count_true = int(out["selected_target_count_true"])
                out["selected_target_count_label_true"] = f"{count_true}/{selection_k}"
                out["selected_target_binomial_tail_p_true"] = _binomial_tail_ge(
                    count_true,
                    selection_k,
                    prevalence_true,
                )
            if out.get("selected_target_count_oracle") is not None:
                count_oracle = int(out["selected_target_count_oracle"])
                out["selected_target_count_label_oracle"] = f"{count_oracle}/{selection_k}"
                out["selected_target_binomial_tail_p_oracle"] = _binomial_tail_ge(
                    count_oracle,
                    selection_k,
                    prevalence_oracle,
                )
        enriched.append(out)
    return enriched


def gate_results_from_metrics(metrics: Sequence[Mapping[str, Any]], safety: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build operator-facing gate rows without changing decision semantics."""
    rows: list[dict[str, Any]] = []
    for key, label in (
        ("path_safety_pass", "H-SAFE"),
        ("forbidden_input_pass", "H-SOURCE"),
        ("x_surface_pass", "H-X-SURFACE"),
    ):
        passed = bool(safety.get(key))
        rows.append(
            {
                "gate": label,
                "status": "pass" if passed else "fail",
                "observed": safety.get(key),
                "threshold": True,
                "reason": f"{key} must be true",
            }
        )
    if any(row.get("status") == "missing_predictions" for row in metrics):
        rows.append(
            {
                "gate": "H-METRICS",
                "status": "fail",
                "reason": "scored runs must fail before decision if prediction metrics are missing",
            }
        )
        return rows
    evaluable = [dict(row) for row in metrics if row.get("status") != "missing_predictions"]
    if not evaluable:
        rows.append({"gate": "H-SCORED", "status": "pending", "reason": "no scored OPAL run metrics"})
        return rows

    by_pair: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in evaluable:
        campaign = str(row.get("campaign") or "")
        split_id = str(row.get("split_id") or "")
        if not campaign or not split_id:
            continue
        oracle_kind = "null" if row.get("oracle_id") == NULL_ORACLE_ID else "positive"
        by_pair.setdefault((campaign, split_id), {})[oracle_kind] = row
    for (campaign, split_id), pair in sorted(by_pair.items()):
        positive = pair.get("positive")
        null = pair.get("null")
        positive_lift = _finite_metric(positive or {}, "target_lift_at_k_true")
        null_lift = _finite_metric(null or {}, "target_lift_at_k_true")
        delta = None if positive_lift is None or null_lift is None else float(positive_lift - null_lift)
        rows.append(
            {
                "gate": "H-NULL-CONTROL",
                "status": (
                    "debug"
                    if null is None
                    else "fail"
                    if null_lift is not None and null_lift > NULL_LIFT_STOP_THRESHOLD
                    else "pass"
                ),
                "campaign": campaign,
                "split_id": split_id,
                "positive_run_key": positive.get("run_key") if positive else None,
                "null_run_key": null.get("run_key") if null else None,
                "positive_lift": positive_lift,
                "null_lift": null_lift,
                "positive_minus_null_lift": delta,
                "null_lift_threshold": NULL_LIFT_STOP_THRESHOLD,
                "reason": (
                    "positive/null pair incomplete"
                    if null is None
                    else f"null lift exceeds {NULL_LIFT_STOP_THRESHOLD:g}"
                    if null_lift is not None and null_lift > NULL_LIFT_STOP_THRESHOLD
                    else "null lift within threshold"
                ),
            }
        )
        rows.append(
            {
                "gate": "H-POSITIVE-SEPARATION",
                "status": (
                    "debug"
                    if positive is None or null is None
                    else "debug"
                    if delta is not None and delta <= 0.0
                    else "pass"
                ),
                "campaign": campaign,
                "split_id": split_id,
                "positive_run_key": positive.get("run_key") if positive else None,
                "null_run_key": null.get("run_key") if null else None,
                "positive_lift": positive_lift,
                "null_lift": null_lift,
                "positive_minus_null_lift": delta,
                "reason": (
                    "positive/null pair incomplete"
                    if positive is None or null is None
                    else "positive lift does not exceed null lift"
                    if delta is not None and delta <= 0.0
                    else "positive lift exceeds paired null"
                ),
            }
        )
    return rows


def decision_reasons_from_metrics(
    metrics: Sequence[Mapping[str, Any]],
    safety: Mapping[str, Any],
    *,
    decision: str | None = None,
) -> list[dict[str, Any]]:
    gate_results = gate_results_from_metrics(metrics, safety)
    reasons = [row for row in gate_results if row.get("status") in {"fail", "debug", "pending"}]
    if decision is not None and not reasons:
        reasons.append(
            {
                "gate": "decision",
                "status": "pass",
                "decision": decision,
                "reason": "all scoped gates passed",
            }
        )
    return [dict(row) for row in reasons]


def metric_quality_from_metrics(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    enriched = enrich_metric_rows(metrics)
    rows = [row for row in enriched if row.get("target_lift_at_k_true") is not None]
    weak_runs = []
    for row in rows:
        p_value = row.get("selected_target_binomial_tail_p_true")
        if p_value is None:
            continue
        try:
            p_float = float(p_value)
        except (TypeError, ValueError):
            continue
        if p_float > 0.05:
            weak_runs.append(
                {
                    "run_key": row.get("run_key"),
                    "selected_target_count": row.get("selected_target_count_label_true"),
                    "p_value": p_float,
                }
            )
    return {
        "run_count": len(rows),
        "weak_count_approx_binomial_p_gt_0_05": len(weak_runs),
        "weak_runs": weak_runs,
    }


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _pass_decision_for_coverage(metrics: Sequence[Mapping[str, Any]]) -> str:
    campaigns = {str(row.get("campaign")) for row in metrics}
    splits = {str(row.get("split_id")) for row in metrics}
    pairs = {(str(row.get("campaign")), str(row.get("split_id"))) for row in metrics}
    all_campaigns = {"cipro", "ethanol", "dual"}
    if campaigns == {"cipro"} and splits == {"random_id"}:
        return PASS_CIPRO_RANDOM_GATE
    if all_campaigns.issubset(campaigns) and splits == {"random_id"}:
        return PASS_RANDOM_ALL_GATE
    if all_campaigns.issubset(campaigns) and splits == {"leave_sigma35_variant"}:
        return PASS_LEAVE_SIGMA35_GATE
    gate_splits = ("random_id", "leave_sigma35_variant")
    required_pairs = {(campaign, split_id) for campaign in all_campaigns for split_id in gate_splits}
    if required_pairs.issubset(pairs):
        return PASS_FULL_MATRIX_GATE
    return PASS_SCOPED_GATE


def _claim_statuses(metrics: list[dict[str, Any]], *, decision: str) -> dict[str, str]:
    if decision == "PENDING":
        deferred = "not evaluated until OPAL run metrics exist"
        return {
            "H-NULL": deferred,
            "H-CIPRO": deferred,
            "H-ETHANOL": deferred,
            "H-DUAL": deferred,
            "H-SIGMA35": deferred,
            "H-COLLAPSE": deferred,
        }
    evaluable = [
        row
        for row in metrics
        if row.get("status") != "missing_predictions" and row.get("target_lift_at_k_true") is not None
    ]
    campaigns = {str(row.get("campaign")) for row in evaluable}
    splits = {str(row.get("split_id")) for row in evaluable}
    has_null = any(row.get("oracle_id") == NULL_ORACLE_ID for row in evaluable)
    has_selection = any(row.get("selected_ids") for row in evaluable)
    return {
        "H-NULL": "evaluated" if has_null else "not evaluated in this run",
        "H-CIPRO": "evaluated" if "cipro" in campaigns else "not evaluated in this run",
        "H-ETHANOL": "evaluated" if "ethanol" in campaigns else "not evaluated in this run",
        "H-DUAL": "evaluated" if "dual" in campaigns else "not evaluated in this run",
        "H-SIGMA35": "evaluated" if "leave_sigma35_variant" in splits else "not evaluated in this run",
        "H-COLLAPSE": "evaluated" if has_selection else "not evaluated in this run",
    }


def _write_decision(
    *,
    path: Path,
    decision: str,
    safety: Mapping[str, Any],
    metrics: list[dict[str, Any]],
    quality_counts: Mapping[str, int],
) -> None:
    key_numbers = {
        "path_safety_pass": safety.get("path_safety_pass"),
        "forbidden_input_pass": safety.get("forbidden_input_pass"),
        "quality_ok_fraction": safety.get("quality_ok_fraction"),
    }
    for row in metrics:
        oracle_kind = "null" if row.get("oracle_id") == NULL_ORACLE_ID else "positive"
        key = f"{row.get('campaign')}_{oracle_kind}_{row.get('split_id')}_target_lift"
        key_numbers[key] = row.get("target_lift_at_k_true")
    claims_heading = "Claims tracked" if decision == "PENDING" else "Claims tested"
    claim_statuses = _claim_statuses(metrics, decision=decision)
    campaign_claims = [
        ("H-NULL", "permuted null did not enrich target classes"),
        ("H-CIPRO", "cipro campaign enriched cipro_only"),
        ("H-ETHANOL", "ethanol campaign enriched ethanol_only"),
        ("H-DUAL", "AND campaign enriched dual_axis_and"),
        ("H-SIGMA35", "signal survived held-out sigma35 variant or failed informatively"),
        ("H-COLLAPSE", "selection did not pathologically collapse into one sampling pocket"),
    ]
    lines = [
        "# opal_densegen_axis_probe_v0 decision",
        "",
        "## Decision",
        "",
        str(decision),
        "",
        f"## {claims_heading}",
        "",
        "- H-SAFE: synthetic labels stayed scratch-only.",
        "- H-SOURCE: oracle generation used DenseGen part metadata only.",
    ]
    for claim_id, description in campaign_claims:
        suffix = f" ({claim_statuses[claim_id]})."
        lines.append(f"- {claim_id}: {description}{suffix}")
    lines.extend(["", "## Key numbers", ""])
    lines.extend(f"- {key}: {value}" for key, value in key_numbers.items())
    lines.extend(["", "## Label quality flags", ""])
    lines.extend(f"- {key}: {value}" for key, value in quality_counts.items())
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "No OPAL run metrics exist yet; this is a source/materialization status, "
                "not a scoped scored-gate decision."
                if decision == "PENDING"
                else "Generated from scratch probe metrics. Treat DEBUG/STOP conservatively; "
                "inspect metrics.json before expanding synthetic-oracle work."
            ),
            "",
            "## Next action",
            "",
            (
                "Run a campaign gate with OPAL execution when ready."
                if decision == "PENDING"
                else "Use this decision to choose OPAL/LatentDNA debugging, assay stratification, "
                "or an initial-label/round-count follow-up."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
