"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/plot_method.py

Notebook component builders for plot method OPAL analysis notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

from ._support import compact_path, display_name, join_list, mapping, sequence
from .plot_text import capability_text, compact_params, plot_math_description, rounds_text


def build_notebook_plot_card_rows(choice: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build compact evidence rows for the selected plot."""

    entry = mapping(choice.get("entry"))
    manifest = mapping(choice.get("manifest"))
    inputs = [
        item
        for item in sequence(manifest.get("inputs"))
        if isinstance(item, Mapping) and (item.get("path") or item.get("role"))
    ]
    base = choice.get("workdir") or manifest.get("campaign_workdir") or manifest.get("workdir")
    return [
        {"field": "plot", "value": entry.get("name") or manifest.get("name")},
        {"field": "display", "value": choice.get("title") or display_name(entry.get("name") or manifest.get("name"))},
        {"field": "kind", "value": entry.get("kind") or manifest.get("kind")},
        {"field": "status", "value": manifest.get("status")},
        {"field": "freshness", "value": choice.get("freshness") or "unknown"},
        {"field": "capability", "value": capability_text(choice.get("capability"))},
        {"field": "generated", "value": manifest.get("generated_at")},
        {"field": "run", "value": manifest.get("run_id") or "all runs"},
        {"field": "rounds", "value": rounds_text(manifest.get("rounds"))},
        {"field": "media", "value": choice.get("path_label") or compact_path(choice.get("path"), base=base)},
        {"field": "tidy data", "value": choice.get("tidy_label") or compact_path(manifest.get("tidy_csv"), base=base)},
        {
            "field": "source data",
            "value": "; ".join(
                f"{item.get('role') or 'input'}={compact_path(item.get('path'), base=base)}" for item in inputs[:5]
            )
            or "not recorded",
        },
        {"field": "warnings", "value": str(len(sequence(manifest.get("warnings"))))},
    ]


def build_notebook_plot_method_rows(choice: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build plot interpretation and math/data-contract rows."""

    manifest = mapping(choice.get("manifest"))
    metadata = mapping(manifest.get("metadata"))
    decision = _plot_decision_metadata(manifest, metadata=metadata)
    capability = mapping(metadata.get("capability")) or mapping(choice.get("capability"))
    kind = str(choice.get("kind") or manifest.get("kind") or "unknown")
    rows = [
        {
            "section": "reading",
            "detail": str(choice.get("caption") or metadata.get("summary") or "No plot description recorded."),
        },
        {"section": "capability", "detail": capability_text(capability)},
        {"section": "data shape", "detail": str(metadata.get("data_shape") or "not recorded")},
        {"section": "parameters", "detail": compact_params(manifest.get("params"))},
        {"section": "math", "detail": plot_math_description(kind, params=mapping(manifest.get("params")))},
        {"section": "tidy schema", "detail": join_list(metadata.get("tidy_schema"), sep=", ")},
        {"section": "failure modes", "detail": join_list(metadata.get("failure_modes"), sep="; ")},
    ]
    if decision is not None:
        rows[1:1] = [
            {"section": "premise", "detail": decision["premise"]},
            {"section": "decision value", "detail": decision["decision_value"]},
            {"section": "rationale", "detail": decision["rationale"]},
            {"section": "claim boundary", "detail": decision["non_claim_boundary"]},
        ]
    return rows


def build_notebook_plot_method_sections(choice: Mapping[str, Any]) -> dict[str, str]:
    """Build readable accordion sections for the selected plot's method."""

    rows = {str(row["section"]): str(row["detail"]) for row in build_notebook_plot_method_rows(choice)}
    manifest = mapping(choice.get("manifest"))
    metadata = mapping(manifest.get("metadata"))
    capability = mapping(metadata.get("capability")) or mapping(choice.get("capability"))
    title = str(choice.get("title") or display_name(choice.get("name"))).strip()
    kind = str(choice.get("kind") or "unknown").replace("_", " ")
    rounds = rounds_text(choice.get("rounds"))
    freshness = str(choice.get("freshness") or "unknown")
    warnings = int(choice.get("warning_count") or 0)
    sections = {
        "Read": (f"{title} shows a {kind} view for {rounds}. {rows.get('reading', 'No plot description recorded.')}"),
        "Math": rows.get("math", "No math description recorded."),
        "Data contract": (
            f"Input data layer: `{capability.get('data_layer', 'unspecified')}`; "
            f"objective family: `{capability.get('objective_family', 'unknown')}`; "
            f"round behavior: `{capability.get('round_scope', 'unspecified')}`; "
            f"labels: `{capability.get('label_requirement', 'none')}`.\n\n"
            f"Data shape: {rows.get('data shape', 'not recorded')}.\n\n"
            f"Counts and replicates: {_plot_count_and_replicate_text(manifest)}\n\n"
            f"Provenance: {_plot_provenance_text(manifest, base=choice.get('workdir'))}\n\n"
            f"Parameters: {rows.get('parameters', 'not recorded')}.\n\n"
            f"Tidy schema: {rows.get('tidy schema', 'not recorded')}.\n\n"
            f"Failure modes: {rows.get('failure modes', 'not recorded')}.\n\n"
            f"Freshness: `{freshness}`. Warnings: `{warnings}`."
        ),
    }
    if "premise" in rows:
        sections = {
            "Read": sections["Read"],
            "Decision": (
                f"**Premise.** {rows['premise']}\n\n"
                f"**Decision use.** {rows['decision value']}\n\n"
                f"**Rationale.** {rows['rationale']}\n\n"
                f"**Claim boundary.** {rows['claim boundary']}"
            ),
            "Math": sections["Math"],
            "Data contract": sections["Data contract"],
        }
    return sections


def _plot_decision_metadata(
    manifest: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
) -> dict[str, str] | None:
    fields = ("premise", "decision_value", "rationale", "non_claim_boundary")
    values = {field: str(manifest.get(field) or metadata.get(field) or "").strip() for field in fields}
    missing = [field for field, value in values.items() if not value]
    if len(missing) == len(fields) and str(manifest.get("tier") or metadata.get("tier") or "").strip() != "decision":
        return None
    if missing:
        raise ValueError(f"Plot manifest decision metadata is missing required fields: {missing}")
    return values


def _plot_provenance_text(manifest: Mapping[str, Any], *, base: Any | None = None) -> str:
    manifest_path = manifest.get("manifest_path")
    generated_at = manifest.get("generated_at")
    run_id = manifest.get("run_id") or "all runs"
    rounds = rounds_text(manifest.get("rounds"))
    inputs = [
        f"{item.get('role') or 'input'}={compact_path(item.get('path'), base=base)}"
        for item in sequence(manifest.get("inputs"))
        if isinstance(item, Mapping) and (item.get("role") or item.get("path"))
    ]
    manifest_label = (
        f"manifest={compact_path(manifest_path, base=base)}"
        if manifest_path not in (None, "")
        else "manifest=not recorded"
    )
    parts = [
        manifest_label,
        f"generated_at={generated_at or 'not recorded'}",
        f"scope={rounds}, run={run_id}",
    ]
    parts.append("inputs=" + ("; ".join(inputs[:5]) if inputs else "not recorded"))
    return ". ".join(parts) + "."


def _plot_count_and_replicate_text(manifest: Mapping[str, Any]) -> str:
    params = mapping(manifest.get("params"))
    metadata = mapping(manifest.get("metadata"))
    values: list[str] = []
    for source in (manifest, metadata, params):
        for key in (
            "selected_count",
            "selection_count",
            "label_count",
            "labels_count",
            "row_count",
            "candidate_count",
            "reference_n",
            "group_count",
            "sample_n",
            "min_n",
            "top_k",
        ):
            value = source.get(key)
            if value not in (None, "", []):
                values.append(f"{key}={value}")
    replicate_values: list[str] = []
    for key in ("replicate", "replicates", "replicate_column", "replicate_id", "aggregation"):
        value = params.get(key) or metadata.get(key)
        if value not in (None, "", []):
            replicate_values.append(f"{key}={value}")
    tidy_schema = {str(item) for item in sequence(metadata.get("tidy_schema"))}
    if "count" in tidy_schema:
        values.append("tidy count rows carry cohort n")
    if "n" in tidy_schema:
        values.append("tidy n column carries row counts")
    if replicate_values:
        values.extend(replicate_values)
    if values:
        return "; ".join(dict.fromkeys(str(value) for value in values)) + "."
    return "No selected, label, or replicate counts are recorded in this plot manifest."
