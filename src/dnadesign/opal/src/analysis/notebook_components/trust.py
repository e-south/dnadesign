"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/trust.py

Notebook trust, readiness, and non-claim presentation contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

from ._support import mapping, sequence
from .plots import build_notebook_visual_surface_model


def build_notebook_trust_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build compact trust-state rows for first-viewport notebook disclosure."""

    status = mapping(view_model.get("status"))
    progress = mapping(view_model.get("progress"))
    state = mapping(progress.get("state"))
    visual_surface = build_notebook_visual_surface_model(view_model)
    warnings = [
        item
        for item in (*sequence(view_model.get("warnings")), *sequence(progress.get("warnings")))
        if isinstance(item, Mapping)
    ]
    blocking_count = sum(1 for item in warnings if item.get("severity") == "error")
    label_status = mapping(view_model.get("label_source_status"))
    return [
        {"field": "status", "value": status.get("progress_status") or "unknown"},
        {"field": "label readiness", "value": label_source_readiness_label(label_status)},
        {"field": "rounds", "value": status.get("round_count") or 0},
        {"field": "state file", "value": "present" if state.get("exists") else "missing"},
        {
            "field": "review manifests",
            "value": len(mapping(view_model.get("review_manifests"))),
        },
        {"field": "plot media choices", "value": len(visual_surface["choices"])},
        {"field": "missing plot outputs", "value": len(visual_surface["missing_outputs"])},
        {"field": "stale artifacts", "value": len(sequence(view_model.get("stale_artifacts")))},
        {"field": "blocking issues", "value": blocking_count},
    ]


def build_notebook_status_line(view_model: Mapping[str, Any]) -> str:
    """Return a compact human status line for the notebook header."""

    row = {str(item["field"]): item["value"] for item in build_notebook_trust_rows(view_model)}
    return (
        f"Status `{row['status']}` across `{row['rounds']}` rounds. "
        f"`{row['plot media choices']}` plot media choices, `{row['missing plot outputs']}` missing plot outputs, "
        f"`{row['stale artifacts']}` stale artifacts, `{row['blocking issues']}` blocking issues."
    )


def build_notebook_validity_lines(view_model: Mapping[str, Any]) -> list[str]:
    """Build explicit trust-state lines for generated notebooks."""

    return [f"{row['field']}: `{row['value']}`" for row in build_notebook_validity_rows(view_model)]


def build_notebook_validity_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build explicit trust-state rows for generated notebooks."""

    status = mapping(view_model.get("status"))
    progress = mapping(view_model.get("progress"))
    state = mapping(progress.get("state"))
    visual_surface = build_notebook_visual_surface_model(view_model)
    plot_manifests = sequence(view_model.get("plot_manifests"))
    stale = sequence(view_model.get("stale_artifacts"))
    warnings = [
        item
        for item in (*sequence(view_model.get("warnings")), *sequence(progress.get("warnings")))
        if isinstance(item, Mapping)
    ]
    blocking_count = sum(1 for item in warnings if item.get("severity") == "error")
    artifact_garden = mapping(view_model.get("artifact_garden"))
    prune_plan = mapping(artifact_garden.get("prune_plan"))
    review_state = len(mapping(view_model.get("review_manifests")))
    state_text = "present" if state.get("exists") else "missing"
    artifact_schema = artifact_garden.get("schema_version") or "unavailable"
    label_status = mapping(view_model.get("label_source_status"))
    rows = [
        {"field": "Campaign status", "value": status.get("progress_status") or "unknown"},
        {"field": "Label readiness", "value": label_source_readiness_label(label_status)},
        {"field": "Progress schema", "value": progress.get("schema_version") or "missing"},
        {"field": "State file", "value": state_text},
        {"field": "Review manifests", "value": review_state},
        {"field": "Plot manifests", "value": len(plot_manifests)},
        {"field": "Written plot media choices", "value": len(visual_surface["choices"])},
        {"field": "Missing plot outputs", "value": len(visual_surface["missing_outputs"])},
        {"field": "Warnings", "value": len(warnings)},
        {"field": "Stale artifacts", "value": len(stale)},
        {"field": "Artifact garden", "value": artifact_schema},
        {"field": "Prune requires apply", "value": prune_plan.get("requires_apply", True)},
        {"field": "Blocking issues", "value": blocking_count},
    ]
    if label_status.get("error"):
        rows.append({"field": "Label contract", "value": str(label_status["error"])})
    return rows


def build_notebook_distrust_lines(view_model: Mapping[str, Any]) -> list[str]:
    """Build a compact distrust/limitations panel for generated notebooks."""

    return [f"{row['field']}: {row['value']}" for row in build_notebook_distrust_rows(view_model)]


def build_notebook_distrust_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build compact limitation rows for generated notebooks."""

    review_manifests = mapping(view_model.get("review_manifests"))
    visual_surface = build_notebook_visual_surface_model(view_model)
    warnings = sequence(view_model.get("warnings"))
    stale = sequence(view_model.get("stale_artifacts"))
    label_status = mapping(view_model.get("label_source_status"))
    rows = [
        {
            "field": "surface boundary",
            "value": "inspection only; execution and mutation stay in the CLI",
        },
        {
            "field": "producer tools",
            "value": "representation browsers and study benchmark reports stay outside canonical OPAL notebooks",
        },
        {
            "field": "review manifests",
            "value": len(review_manifests),
        },
    ]
    if not visual_surface["choices"]:
        rows.append({"field": "plot evidence", "value": "no plot media"})
    if warnings:
        rows.append({"field": "warnings", "value": len(warnings)})
    if stale:
        rows.append({"field": "stale artifacts ignored by active manifests", "value": len(stale)})
    if label_source_readiness_label(label_status) == "blocked":
        rows.append(
            {
                "field": "label contract",
                "value": str(label_status.get("error") or "Manifest-pinned observed-label source is not verified."),
            }
        )
    return rows


def label_source_readiness_label(status: Mapping[str, Any]) -> str:
    if not status:
        return "not reported"
    if status.get("valid") is True:
        return "ready"
    if status.get("valid") is False or (status.get("manifest_pinned") and status.get("valid") is not True):
        return "blocked"
    return "not verified"


def campaign_claim_boundary(view_model: Mapping[str, Any]) -> str:
    label_status = mapping(view_model.get("label_source_status"))
    if not label_status:
        return ""
    readiness = label_source_readiness_label(label_status)
    status = mapping(view_model.get("status"))
    if readiness == "blocked":
        return "No model or selection claim is available until the observed-label contract verifies."
    if readiness != "ready":
        return "No model or selection claim is available until label readiness verifies."
    if int(status.get("round_count") or 0) == 0:
        return "No model or selection claim is available until a campaign run completes."
    return "Review evidence is scoped to the selected campaign run; synthesis authorization remains study-owned."


def campaign_evidence_status_lines(view_model: Mapping[str, Any]) -> list[str]:
    label_status = mapping(view_model.get("label_source_status"))
    if not label_status:
        return []
    readiness = label_source_readiness_label(label_status)
    boundary = campaign_claim_boundary(view_model)
    if readiness == "blocked":
        lines = [f"**Evidence status:** Blocked. {boundary}"]
        if label_status.get("error"):
            lines.append(f"**Blocking label contract:** {label_status['error']}")
        return lines
    if readiness == "ready":
        return [f"**Evidence status:** Observed-label source verified. {boundary}"]
    return [f"**Evidence status:** Label readiness is not verified. {boundary}"]
