"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/overview.py

Notebook component builders for overview OPAL analysis notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

from ._support import compact_path, display_name, mapping, selection_count, sequence
from .campaign_labels import campaign_dropdown_label
from .plots import build_notebook_visual_surface_model


def build_notebook_campaign_summary_row(view_model: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact campaign row for notebook overview tables."""

    campaign = mapping(view_model.get("campaign"))
    status = mapping(view_model.get("status"))
    stale_count = len(sequence(view_model.get("stale_artifacts")))
    warning_count = len(sequence(view_model.get("warnings")))
    label_context = _campaign_label_context(campaign)
    label = campaign_dropdown_label(
        campaign,
        status=status,
        title=_campaign_title(campaign),
        label_context=label_context,
    )
    return {
        "label": label,
        "campaign": campaign.get("slug"),
        "name": _campaign_title(campaign),
        "status": status.get("progress_status"),
        "round_count": status.get("round_count"),
        "latest_run_id": status.get("latest_run_id"),
        "x_column": campaign.get("x_column"),
        "y_column": campaign.get("y_column"),
        "label_source": campaign.get("label_source"),
        "label_context": label_context,
        "metadata_context": _campaign_metadata_context(campaign),
        "plots": len(sequence(view_model.get("plot_manifests"))),
        "stale": stale_count,
        "warnings": warning_count,
    }


def build_notebook_campaign_header_lines(view_model: Mapping[str, Any], *, heading_level: int = 1) -> list[str]:
    """Build a compact, human-readable notebook heading."""

    campaign = mapping(view_model.get("campaign"))
    title = _campaign_title(campaign)
    level = max(1, min(6, int(heading_level)))
    marker = "#" * level
    description = _campaign_description(campaign)
    return [f"{marker} {title}", "", description]


def _campaign_title(campaign: Mapping[str, Any]) -> str:
    slug = str(campaign.get("slug") or "unknown").strip()
    name = str(campaign.get("name") or "").strip()
    title = name if name and name != slug else display_name(slug)
    for suffix in {slug, slug.removeprefix("opal_axis_probe_v0_")}:
        if suffix and title.endswith(f" [{suffix}]"):
            title = title[: -len(f" [{suffix}]")].strip()
    title = title.replace("top_n", "top N")
    if title.lower().startswith("opal "):
        title = "OPAL " + title[5:]
    return title


def _campaign_description(campaign: Mapping[str, Any]) -> str:
    metadata_description = _campaign_metadata_description(campaign)
    if metadata_description:
        return metadata_description

    description = str(campaign.get("description") or "").strip()
    if description:
        return description

    title = _campaign_title(campaign)
    slug = str(campaign.get("slug") or "unknown").strip()
    objective = sequence(campaign.get("objectives"))[0] if sequence(campaign.get("objectives")) else "objective"
    x_column = str(campaign.get("x_column") or "").strip()
    x_clause = f" The active X contract is `{x_column}`." if x_column else ""
    return (
        f"Campaign ID `{slug}`. {title} scores the configured OPAL records table with `{campaign.get('model')}` "
        f"and selects candidates by `{campaign.get('selection')}` against `{objective}`.{x_clause}"
    )


def _campaign_metadata_description(campaign: Mapping[str, Any]) -> str:
    metadata = mapping(campaign.get("metadata"))
    target = str(
        metadata.get("target_dropdown_label") or metadata.get("target_label") or metadata.get("target") or ""
    ).strip()
    if not target:
        return ""
    role = _metadata_role_label(metadata)
    seed = metadata.get("replicate_seed") if metadata.get("replicate_seed") is not None else metadata.get("seed")
    seed_clause = f", seed {seed}" if seed is not None else ""
    rounds = metadata.get("rounds")
    selection_k = metadata.get("selection_k")
    budget_clause = ""
    if rounds is not None and selection_k is not None:
        budget_clause = f" The selection budget is {rounds} rounds x {selection_k} records."
    return (
        f"Pre-assay metadata probe for {target} using the {role}{seed_clause}. "
        "It tests whether the X representation supports active enrichment for this metadata, "
        f"not measured phenotype prediction.{budget_clause}"
    )


def _metadata_role_label(metadata: Mapping[str, Any]) -> str:
    role = str(metadata.get("label_oracle_kind") or metadata.get("oracle_role") or "").strip()
    labels = {
        "positive": "sequence-matched metadata table",
        "null": "control label table",
        "matched_null": "control label table",
    }
    return labels.get(role, display_name(role) if role else "configured label table")


def build_notebook_at_a_glance_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build first-viewport campaign status rows from a notebook view model."""

    row = build_notebook_campaign_summary_row(view_model)
    campaign = mapping(view_model.get("campaign"))
    status = mapping(view_model.get("status"))
    workdir = campaign.get("workdir")
    selected_count = selection_count(view_model)
    rows = [
        {"field": "campaign", "value": row["campaign"]},
        {"field": "description", "value": _campaign_description(campaign)},
        {"field": "description source", "value": campaign.get("description_source") or "derived"},
        {"field": "status", "value": row["status"]},
        {"field": "round selector", "value": status.get("round_selector")},
        {"field": "round count", "value": row["round_count"]},
        {"field": "latest run", "value": row["latest_run_id"]},
        {"field": "X column", "value": row["x_column"]},
        {"field": "Y column", "value": row["y_column"]},
        {"field": "label source", "value": row["label_source"]},
        {"field": "label context", "value": row["label_context"] or "not recorded"},
        {"field": "campaign metadata", "value": row["metadata_context"] or "not recorded"},
        {"field": "config", "value": compact_path(campaign.get("config_path"), base=workdir)},
        {"field": "workspace", "value": compact_path(workdir, max_parts=1)},
    ]
    if selected_count is not None:
        rows.append({"field": "selected count", "value": selected_count})
    rows.extend(
        (
            {"field": "configured plots", "value": row["plots"]},
            {"field": "warnings", "value": row["warnings"]},
            {"field": "stale artifacts", "value": row["stale"]},
        )
    )
    return rows


def _campaign_label_context(campaign: Mapping[str, Any]) -> str:
    metadata = mapping(campaign.get("metadata"))
    parts = []
    for key in ("label_family_id", "label_oracle_kind", "label_split_id"):
        if metadata.get(key):
            parts.append(f"{key}={metadata[key]}")
    probe_aliases = {
        "probe_label_family_id": "label_family_id",
        "probe_oracle_kind": "label_oracle_kind",
        "probe_split_id": "label_split_id",
    }
    for key, label in probe_aliases.items():
        if metadata.get(key) and not metadata.get(label):
            parts.append(f"{label}={metadata[key]}")
    for key in sorted(metadata):
        if key in {"label_family_id", "label_oracle_kind", "label_split_id", *probe_aliases}:
            continue
        if key.endswith(("_label_family_id", "_oracle_kind", "_split_id")) and metadata.get(key):
            parts.append(f"{key}={metadata[key]}")
    if not parts:
        y_column = str(campaign.get("y_column") or "").strip()
        if y_column:
            parts.append(f"y={y_column}")
    return "; ".join(parts[:4])


def _campaign_metadata_context(campaign: Mapping[str, Any]) -> str:
    metadata = mapping(campaign.get("metadata"))
    parts = []
    for key in ("campaign_context", "campaign_kind", "campaign_type", "study_id", "response_axis", "comparison_group"):
        if metadata.get(key):
            parts.append(f"{key}={metadata[key]}")
    return "; ".join(parts[:6])


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
    return [
        {"field": "status", "value": status.get("progress_status") or "unknown"},
        {"field": "rounds", "value": status.get("round_count") or 0},
        {"field": "state file", "value": "present" if state.get("exists") else "missing"},
        {
            "field": "review manifest",
            "value": "present" if isinstance(view_model.get("review_manifest"), Mapping) else "missing",
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
    review_state = "present" if isinstance(view_model.get("review_manifest"), Mapping) else "missing"
    state_text = "present" if state.get("exists") else "missing"
    artifact_schema = artifact_garden.get("schema_version") or "unavailable"
    return [
        {"field": "Campaign status", "value": status.get("progress_status") or "unknown"},
        {"field": "Progress schema", "value": progress.get("schema_version") or "missing"},
        {"field": "State file", "value": state_text},
        {"field": "Review manifest", "value": review_state},
        {"field": "Plot manifests", "value": len(plot_manifests)},
        {"field": "Written plot media choices", "value": len(visual_surface["choices"])},
        {"field": "Missing plot outputs", "value": len(visual_surface["missing_outputs"])},
        {"field": "Warnings", "value": len(warnings)},
        {"field": "Stale artifacts", "value": len(stale)},
        {"field": "Artifact garden", "value": artifact_schema},
        {"field": "Prune requires apply", "value": prune_plan.get("requires_apply", True)},
        {"field": "Blocking issues", "value": blocking_count},
    ]


def build_notebook_distrust_lines(view_model: Mapping[str, Any]) -> list[str]:
    """Build a compact distrust/limitations panel for generated notebooks."""

    return [f"{row['field']}: {row['value']}" for row in build_notebook_distrust_rows(view_model)]


def build_notebook_distrust_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build compact limitation rows for generated notebooks."""

    review_manifest = view_model.get("review_manifest")
    visual_surface = build_notebook_visual_surface_model(view_model)
    warnings = sequence(view_model.get("warnings"))
    stale = sequence(view_model.get("stale_artifacts"))
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
            "field": "review manifest",
            "value": "missing" if review_manifest is None else "present",
        },
    ]
    if not visual_surface["choices"]:
        rows.append({"field": "plot evidence", "value": "no plot media"})
    if warnings:
        rows.append({"field": "warnings", "value": len(warnings)})
    if stale:
        rows.append({"field": "stale artifacts ignored by active manifests", "value": len(stale)})
    return rows
