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
from .trust import campaign_claim_boundary, campaign_evidence_status_lines, label_source_readiness_label


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


def build_notebook_campaign_header_lines(
    view_model: Mapping[str, Any],
    *,
    selection_view: Mapping[str, Any],
    heading_level: int = 1,
) -> list[str]:
    """Build a compact, human-readable notebook heading."""

    campaign = mapping(view_model.get("campaign"))
    title = _campaign_title(campaign)
    level = max(1, min(6, int(heading_level)))
    marker = "#" * level
    description = _campaign_description(campaign)
    target = _objective_target_summary(selection_view, campaign=campaign)
    lines = [f"{marker} {title}", "", description]
    evidence_lines = campaign_evidence_status_lines(view_model)
    if evidence_lines:
        lines.extend(("", *evidence_lines))
    lines.extend(("", f"**Objective target:** {target}."))
    return lines


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
    selection_views = sequence(campaign.get("selection_views"))
    if not selection_views:
        raise ValueError("Campaign notebook description requires at least one selection view.")
    view_label = "selection view" if len(selection_views) == 1 else "selection views"
    x_column = str(campaign.get("x_column") or "").strip()
    x_clause = f" The active X contract is `{x_column}`." if x_column else ""
    return (
        f"Campaign ID `{slug}`. {title} fits `{campaign.get('model')}` once and evaluates "
        f"{len(selection_views)} {view_label} from the shared predictions.{x_clause}"
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


def build_notebook_at_a_glance_rows(
    view_model: Mapping[str, Any],
    *,
    selection_view: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build first-viewport campaign status rows from a notebook view model."""

    row = build_notebook_campaign_summary_row(view_model)
    campaign = mapping(view_model.get("campaign"))
    status = mapping(view_model.get("status"))
    workdir = campaign.get("workdir")
    selected_count = selection_count(view_model)
    label_status = mapping(view_model.get("label_source_status"))
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
        {"field": "selection view", "value": selection_view.get("id")},
        {
            "field": "objective target",
            "value": _objective_target_summary(selection_view, campaign=campaign),
        },
        {"field": "label source", "value": row["label_source"]},
        {"field": "label readiness", "value": label_source_readiness_label(label_status)},
        {"field": "label context", "value": row["label_context"] or "not recorded"},
        {"field": "campaign metadata", "value": row["metadata_context"] or "not recorded"},
        {"field": "config", "value": compact_path(campaign.get("config_path"), base=workdir)},
        {"field": "workspace", "value": compact_path(workdir, max_parts=1)},
    ]
    if label_status.get("error"):
        rows.append({"field": "label contract", "value": str(label_status["error"])})
    claim_boundary = campaign_claim_boundary(view_model)
    if claim_boundary:
        rows.append({"field": "claim boundary", "value": claim_boundary})
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


def _objective_target_summary(
    selection_view: Mapping[str, Any],
    *,
    campaign: Mapping[str, Any] | None = None,
) -> str:
    view_id = str(selection_view.get("id") or "").strip()
    if not view_id:
        raise ValueError("Notebook selection view requires a non-empty id.")
    objective = mapping(selection_view.get("objective"))
    name = str(objective.get("name") or "").strip()
    if not name:
        raise ValueError(f"Notebook selection view {view_id!r} requires an objective name.")
    params = mapping(objective.get("params"))
    if "state_ids" in params or "target_mask" in params:
        state_ids = [str(value) for value in sequence(params.get("state_ids"))]
        target_mask = sequence(params.get("target_mask"))
        if not state_ids or len(state_ids) != len(target_mask):
            raise ValueError("Masked notebook objectives require aligned state_ids and target_mask values.")
        if any(isinstance(value, bool) or value not in (0, 1) for value in target_mask):
            raise ValueError("Masked notebook objective target_mask values must be numeric zero or one.")
        on_states = [state for state, value in zip(state_ids, target_mask, strict=True) if int(value) == 1]
        off_states = [state for state, value in zip(state_ids, target_mask, strict=True) if int(value) == 0]
        if not on_states or not off_states:
            raise ValueError("Masked notebook objective target_mask must contain at least one ON and one OFF state.")
        objective_label = display_name(name.removesuffix("_v1"))
        acronym = _campaign_objective_acronym(campaign, objective_name=name)
        if acronym:
            objective_label = f"{objective_label} ({acronym})"
        selection = mapping(selection_view.get("selection"))
        selection_params = mapping(selection.get("params"))
        score_ref = str(selection_params.get("score_ref") or "").strip()
        objective_mode = str(selection_params.get("objective_mode") or "").strip().lower()
        score_clause = (
            f"; {objective_mode} {display_name(score_ref).lower()}"
            if score_ref and objective_mode in {"maximize", "minimize"}
            else ""
        )
        return f"{objective_label}{score_clause}; ON={', '.join(on_states)}; OFF={', '.join(off_states)}"
    if "setpoint_vector" in params:
        return f"{name} setpoint_vector={params['setpoint_vector']}"
    if params:
        return f"{name} params={sorted(params)}"
    return f"{name} params=none"


def _campaign_objective_acronym(
    campaign: Mapping[str, Any] | None,
    *,
    objective_name: str,
) -> str:
    campaign_record = mapping(campaign)
    acronym = str(mapping(campaign_record.get("metadata")).get("metric_acronym") or "").strip()
    if not acronym:
        return ""
    objective_names = {
        str(mapping(mapping(view).get("objective")).get("name") or "").strip()
        for view in sequence(campaign_record.get("selection_views"))
    }
    objective_names.discard("")
    return acronym if objective_names == {objective_name} else ""


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
