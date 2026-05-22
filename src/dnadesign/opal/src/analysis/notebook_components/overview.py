from __future__ import annotations

from typing import Any, Mapping

from ._support import compact_path, join_list, mapping, selection_count, sequence
from .plots import build_notebook_visual_surface_model


def build_notebook_campaign_summary_row(view_model: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact campaign row for notebook overview tables."""

    campaign = mapping(view_model.get("campaign"))
    status = mapping(view_model.get("status"))
    stale_count = len(sequence(view_model.get("stale_artifacts")))
    warning_count = len(sequence(view_model.get("warnings")))
    label = f"{campaign.get('slug') or 'unknown'} | {status.get('progress_status') or 'unknown'}"
    return {
        "label": label,
        "campaign": campaign.get("slug"),
        "status": status.get("progress_status"),
        "round_count": status.get("round_count"),
        "latest_run_id": status.get("latest_run_id"),
        "x_column": campaign.get("x_column"),
        "label_source": campaign.get("label_source"),
        "plots": len(sequence(view_model.get("plot_manifests"))),
        "stale": stale_count,
        "warnings": warning_count,
    }


def build_notebook_at_a_glance_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build first-viewport campaign status rows from a notebook view model."""

    row = build_notebook_campaign_summary_row(view_model)
    campaign = mapping(view_model.get("campaign"))
    status = mapping(view_model.get("status"))
    workdir = campaign.get("workdir")
    selected_count = selection_count(view_model)
    rows = [
        {"field": "campaign", "value": row["campaign"]},
        {"field": "status", "value": row["status"]},
        {"field": "round selector", "value": status.get("round_selector")},
        {"field": "round count", "value": row["round_count"]},
        {"field": "latest run", "value": row["latest_run_id"]},
        {"field": "X column", "value": row["x_column"]},
        {"field": "label source", "value": row["label_source"]},
        {"field": "config", "value": compact_path(campaign.get("config_path"), base=workdir)},
        {"field": "workspace", "value": compact_path(workdir, max_parts=1)},
    ]
    if selected_count is not None:
        rows.append({"field": "selected count", "value": selected_count})
    rows.extend(
        (
            {"field": "manifest-backed plots", "value": row["plots"]},
            {"field": "warnings", "value": row["warnings"]},
            {"field": "stale artifacts", "value": row["stale"]},
        )
    )
    return rows


def build_notebook_validity_lines(view_model: Mapping[str, Any]) -> list[str]:
    """Build explicit trust-state lines for generated notebooks."""

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
    review_state = "present" if isinstance(view_model.get("review_manifest"), Mapping) else "missing"
    state_text = "present" if state.get("exists") else "missing"
    artifact_schema = artifact_garden.get("schema_version") or "unavailable"
    return [
        f"- Campaign status: `{status.get('progress_status') or 'unknown'}`",
        f"- Progress schema: `{progress.get('schema_version') or 'missing'}`",
        f"- State file: `{state_text}`",
        f"- Review manifest: `{review_state}`",
        f"- Plot manifests: `{len(plot_manifests)}`",
        f"- Written plot media choices: `{len(visual_surface['choices'])}`",
        f"- Missing plot outputs: `{len(visual_surface['missing_outputs'])}`",
        f"- Warnings: `{len(warnings)}`",
        f"- Stale artifacts: `{len(stale)}`",
        f"- Artifact garden: `{artifact_schema}`",
        f"- Prune requires apply: `{prune_plan.get('requires_apply', True)}`",
        f"- Blocking issues: `{blocking_count}`",
    ]


def build_notebook_distrust_lines(view_model: Mapping[str, Any]) -> list[str]:
    """Build a compact distrust/limitations panel for generated notebooks."""

    review_manifest = view_model.get("review_manifest")
    visual_surface = build_notebook_visual_surface_model(view_model)
    warnings = sequence(view_model.get("warnings"))
    stale = sequence(view_model.get("stale_artifacts"))
    lines = [
        "- OPAL notebooks are inspection surfaces; execution and mutation stay in the CLI.",
        "- Producer-specific representation browsers and study benchmark reports are outside this notebook.",
    ]
    lines.append("- Review manifest: `missing`" if review_manifest is None else "- Review manifest: `present`")
    if not visual_surface["choices"]:
        lines.append("- Plot evidence: no written manifest-backed plot media.")
    if warnings:
        lines.append(f"- Warnings: `{len(warnings)}`")
    if stale:
        lines.append(f"- Stale artifacts ignored by active manifests: `{len(stale)}`")
    return lines


def build_notebook_evidence_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return warning and stale-artifact rows for notebook evidence tables."""

    campaign = mapping(view_model.get("campaign"))
    workdir = campaign.get("workdir")
    rows: list[dict[str, Any]] = []
    for label, path in (
        ("config", campaign.get("config_path")),
        ("workdir", workdir),
        ("records", campaign.get("records_path")),
        ("review_manifest", view_model.get("review_manifest_path")),
    ):
        if path:
            rows.append(
                {
                    "source": "path",
                    "category": label,
                    "severity": None,
                    "message": compact_path(path, base=workdir),
                    "path": compact_path(path, base=workdir),
                }
            )
    for warning in sequence(view_model.get("warnings")):
        if isinstance(warning, Mapping):
            rows.append(
                {
                    "source": "warning",
                    "category": warning.get("category"),
                    "severity": warning.get("severity"),
                    "message": warning.get("message"),
                    "path": compact_path(warning.get("path"), base=workdir),
                }
            )
    for artifact in sequence(view_model.get("stale_artifacts")):
        if isinstance(artifact, Mapping):
            rows.append(
                {
                    "source": "stale_artifact",
                    "category": artifact.get("category"),
                    "severity": artifact.get("severity"),
                    "message": artifact.get("message"),
                    "path": compact_path(artifact.get("path"), base=workdir),
                }
            )
    return rows


def build_notebook_metric_definition_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return plot metric/data-shape definitions for notebook evidence tables."""

    rows: list[dict[str, Any]] = []
    for manifest in sequence(view_model.get("plot_manifests")):
        if not isinstance(manifest, Mapping):
            continue
        metadata = mapping(manifest.get("metadata"))
        freshness = mapping(manifest.get("freshness"))
        purpose = manifest.get("review_purpose") or manifest.get("caption") or metadata.get("summary") or "not recorded"
        rows.append(
            {
                "plot": manifest.get("name"),
                "kind": manifest.get("kind"),
                "data_shape": metadata.get("data_shape") or "not recorded",
                "tidy_schema": join_list(metadata.get("tidy_schema"), sep=", "),
                "failure_modes": join_list(metadata.get("failure_modes"), sep="; "),
                "freshness": freshness.get("status") or "unknown",
                "purpose": purpose,
            }
        )
    return rows
