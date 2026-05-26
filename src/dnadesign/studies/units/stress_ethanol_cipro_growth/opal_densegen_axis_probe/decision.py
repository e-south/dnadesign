"""Study-owned DenseGen axis OPAL probe package."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import NULL_ORACLE_ID, ORACLE_ID
from .round_dynamics import (
    round_dynamics_from_metrics,
    round_gate_results_from_metrics,
)
from .trajectory_metrics import (
    has_trajectory_separation_failure,
    trajectory_gate_results_from_metrics,
    trajectory_metric_payload,
)

PASS_CIPRO_RANDOM_GATE = "PASS_CIPRO_RANDOM_GATE"
PASS_RANDOM_ALL_GATE = "PASS_RANDOM_ALL_GATE"
PASS_LEAVE_SIGMA35_GATE = "PASS_LEAVE_SIGMA35_GATE"
PASS_FULL_MATRIX_GATE = "PASS_FULL_MATRIX_GATE"
PASS_SCOPED_GATE = "PASS_SCOPED_GATE"
NULL_LIFT_ATTENTION_BASELINE = 1.0


def metric_definitions() -> dict[str, str]:
    return {
        "selected_target_count": (
            "For DenseGen plan-logic4, how many selected candidates match the run target class. For TF-count "
            "probes, how many selected candidates have a nonzero value for the active count objective."
        ),
        "precision_at_k": "selected_target_count divided by K.",
        "target_prevalence": (
            "For DenseGen plan-logic4, fraction of evaluable candidates in the split pool that match the target "
            "class. For TF-count probes, fraction with nonzero target count."
        ),
        "lift": (
            "For DenseGen plan-logic4, lift = precision@K / target prevalence. For TF-count probes, lift = "
            "selected mean count / evaluable-pool mean count. Values above 1 mean enrichment over random selection."
        ),
        "binomial_tail_p": (
            "Approximate probability of seeing at least the observed selected_target_count under random selection "
            "from a pool with the same target prevalence."
        ),
        "null_lift": (
            "The same lift calculation for a campaign trained on permuted labels, still evaluated against the true "
            "target class. Null lift is diagnostic; paired positive-vs-null separation decides QA status."
        ),
        "trajectory_auc": (
            "Round-normalized area under the target-lift trajectory. With one scored point, the AUC is that point."
        ),
        "paired_auc_delta": (
            "positive trajectory AUC minus paired null trajectory AUC for the same campaign/split/seed."
        ),
        "round": (
            "Round metrics are computed retroactively from each OPAL prediction ledger. Round 0 uses the initial "
            "labels; later rounds include the previous round's selected labels."
        ),
        "round_dynamics": (
            "Per-run first/final/max lift over OPAL rounds. Null-control spikes above random-baseline lift are "
            "non-blocking attention signals."
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


def _decision_from_metrics(
    metrics: list[dict[str, Any]],
    safety: Mapping[str, Any],
    *,
    round_metrics: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    if not safety.get("path_safety_pass") or not safety.get("forbidden_input_pass") or not safety.get("x_surface_pass"):
        return "STOP"
    if not metrics:
        return "PENDING"
    if any(row.get("status") == "missing_predictions" for row in metrics):
        raise ValueError("metrics contain missing_predictions; scored runs must fail before decision")
    evaluable = [row for row in metrics if row.get("status") != "missing_predictions"]
    if not evaluable:
        return "DEBUG"

    pair_keys = sorted(
        {
            (str(row.get("label_family_id") or "unknown"), str(row.get("campaign")), str(row.get("split_id")))
            for row in evaluable
        }
    )
    if not pair_keys:
        return "DEBUG"
    for label_family_id, campaign, split_id in pair_keys:
        positive = [
            value
            for row in evaluable
            if str(row.get("label_family_id") or "unknown") == label_family_id
            and row.get("campaign") == campaign
            and row.get("split_id") == split_id
            and row.get("oracle_id") == ORACLE_ID
            for value in [_finite_metric(row, "target_lift_at_k_true")]
            if value is not None
        ]
        null = [
            value
            for row in evaluable
            if str(row.get("label_family_id") or "unknown") == label_family_id
            and row.get("campaign") == campaign
            and row.get("split_id") == split_id
            and row.get("oracle_id") == NULL_ORACLE_ID
            for value in [_finite_metric(row, "target_lift_at_k_true")]
            if value is not None
        ]
        if not positive or not null:
            return "DEBUG"
        if max(positive) <= max(null):
            return "DEBUG"
    if round_metrics and has_trajectory_separation_failure(run_metrics=evaluable, round_metrics=round_metrics):
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


def gate_results_from_metrics(
    metrics: Sequence[Mapping[str, Any]],
    safety: Mapping[str, Any],
    *,
    round_metrics: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
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

    by_pair: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in evaluable:
        label_family_id = str(row.get("label_family_id") or "unknown")
        campaign = str(row.get("campaign") or "")
        split_id = str(row.get("split_id") or "")
        if not campaign or not split_id:
            continue
        oracle_kind = "null" if row.get("oracle_id") == NULL_ORACLE_ID else "positive"
        by_pair.setdefault((label_family_id, campaign, split_id), {})[oracle_kind] = row
    for (label_family_id, campaign, split_id), pair in sorted(by_pair.items()):
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
                    else "attention"
                    if null_lift is not None and null_lift > NULL_LIFT_ATTENTION_BASELINE
                    else "pass"
                ),
                "campaign": campaign,
                "label_family_id": label_family_id,
                "split_id": split_id,
                "positive_run_key": positive.get("run_key") if positive else None,
                "null_run_key": null.get("run_key") if null else None,
                "positive_lift": positive_lift,
                "null_lift": null_lift,
                "positive_minus_null_lift": delta,
                "null_lift_attention_baseline": NULL_LIFT_ATTENTION_BASELINE,
                "reason": (
                    "positive/null pair incomplete"
                    if null is None
                    else "null lift exceeds random-baseline lift; diagnostic only"
                    if null_lift is not None and null_lift > NULL_LIFT_ATTENTION_BASELINE
                    else "null lift recorded for paired QA comparison"
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
                "label_family_id": label_family_id,
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
    if round_metrics:
        rows.extend(round_gate_results_from_metrics(round_metrics, null_lift_threshold=NULL_LIFT_ATTENTION_BASELINE))
        rows.extend(trajectory_gate_results_from_metrics(run_metrics=evaluable, round_metrics=round_metrics))
    return rows


def decision_reasons_from_metrics(
    metrics: Sequence[Mapping[str, Any]],
    safety: Mapping[str, Any],
    *,
    decision: str | None = None,
    round_metrics: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    gate_results = gate_results_from_metrics(metrics, safety, round_metrics=round_metrics)
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


def round_dynamics_summary(round_metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return round_dynamics_from_metrics(round_metrics, null_lift_threshold=NULL_LIFT_ATTENTION_BASELINE)


def trajectory_qa_summary(
    metrics: Sequence[Mapping[str, Any]],
    round_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return trajectory_metric_payload(run_metrics=metrics, round_metrics=round_metrics)


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
    families = {str(row.get("label_family_id") or "unknown") for row in metrics}
    pairs = {
        (str(row.get("label_family_id") or "unknown"), str(row.get("campaign")), str(row.get("split_id")))
        for row in metrics
    }
    all_campaigns = {"cipro", "ethanol", "dual"}
    if campaigns == {"cipro"} and splits == {"random_id"}:
        return PASS_CIPRO_RANDOM_GATE
    if all_campaigns.issubset(campaigns) and splits == {"random_id"}:
        return PASS_RANDOM_ALL_GATE
    if all_campaigns.issubset(campaigns) and splits == {"leave_sigma35_variant"}:
        return PASS_LEAVE_SIGMA35_GATE
    gate_splits = ("random_id", "leave_sigma35_variant")
    required_pairs = {
        (family, campaign, split_id) for family in families for campaign in all_campaigns for split_id in gate_splits
    }
    if required_pairs.issubset(pairs):
        return PASS_FULL_MATRIX_GATE
    return PASS_SCOPED_GATE
