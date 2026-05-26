"""Decision-report writing for the DenseGen axis probe."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .constants import NULL_ORACLE_ID


def write_decision(
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
        key = f"{row.get('label_family_id')}_{row.get('campaign')}_{oracle_kind}_{row.get('split_id')}_target_lift"
        key_numbers[key] = row.get("target_lift_at_k_true")
    lines = _decision_lines(
        decision=decision,
        metrics=metrics,
        key_numbers=key_numbers,
        quality_counts=quality_counts,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _decision_lines(
    *,
    decision: str,
    metrics: list[dict[str, Any]],
    key_numbers: Mapping[str, Any],
    quality_counts: Mapping[str, int],
) -> list[str]:
    claims_heading = "Claims tracked" if decision == "PENDING" else "Claims tested"
    claim_statuses = _claim_statuses(metrics, decision=decision)
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
    for claim_id, description in _campaign_claims():
        lines.append(f"- {claim_id}: {description} ({claim_statuses[claim_id]}).")
    lines.extend(["", "## Key numbers", ""])
    lines.extend(f"- {key}: {value}" for key, value in key_numbers.items())
    lines.extend(["", "## Label quality flags", ""])
    lines.extend(f"- {key}: {value}" for key, value in quality_counts.items())
    lines.extend(["", "## Interpretation", "", _interpretation_text(decision), "", "## Next action", ""])
    lines.extend([_next_action_text(decision), ""])
    return lines


def _campaign_claims() -> tuple[tuple[str, str], ...]:
    return (
        ("H-NULL", "permuted null did not enrich target classes"),
        ("H-CIPRO", "cipro campaign enriched cipro_only"),
        ("H-ETHANOL", "ethanol campaign enriched ethanol_only"),
        ("H-DUAL", "AND campaign enriched dual_axis_and"),
        ("H-SIGMA35", "signal survived held-out sigma35 variant or failed informatively"),
        ("H-COLLAPSE", "selection did not pathologically collapse into one sampling pocket"),
    )


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


def _interpretation_text(decision: str) -> str:
    if decision == "PENDING":
        return (
            "No OPAL run metrics exist yet; this is a source/materialization status, not a scoped scored-gate decision."
        )
    return (
        "Generated from scratch probe metrics. Treat DEBUG/STOP conservatively; "
        "inspect metrics.json before expanding synthetic-oracle work."
    )


def _next_action_text(decision: str) -> str:
    if decision == "PENDING":
        return "Run a campaign gate with OPAL execution when ready."
    return (
        "Use this decision to choose OPAL/LatentDNA debugging, assay stratification, "
        "or an initial-label/round-count follow-up."
    )
