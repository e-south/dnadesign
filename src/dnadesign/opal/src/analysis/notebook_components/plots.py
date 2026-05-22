from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from ._support import (
    compact_path,
    display_name,
    first_media_output,
    join_list,
    mapping,
    plot_entries_from_manifests,
    sequence,
)


def build_notebook_visual_surface_model(
    view_model: Mapping[str, Any],
    *,
    plot_entries: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build manifest-authoritative visual choices for OPAL marimo templates."""

    campaign = mapping(view_model.get("campaign"))
    workdir = campaign.get("workdir") or ""
    plots_dir = str(Path(str(workdir)) / "outputs" / "plots") if workdir else "outputs/plots"
    manifest_rows = [
        manifest
        for manifest in sequence(view_model.get("plot_manifests"))
        if isinstance(manifest, Mapping) and manifest.get("status") == "written"
    ]
    active_by_name = {str(row.get("name")): row for row in manifest_rows}
    configured_entries = plot_entries_from_manifests(manifest_rows) if plot_entries is None else list(plot_entries)

    choices: list[dict[str, Any]] = []
    missing_outputs: list[str] = []
    labels_seen: dict[str, int] = {}
    for entry in configured_entries:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "")
        if not name:
            continue
        manifest = active_by_name.get(name)
        if manifest is None:
            missing_outputs.append(name)
            continue
        media_output = first_media_output(manifest)
        if media_output is None:
            missing_outputs.append(name)
            continue
        path = str(media_output.get("path"))
        title = str(entry.get("title") or manifest.get("title") or display_name(name))
        label = title
        labels_seen[label] = labels_seen.get(label, 0) + 1
        if labels_seen[label] > 1:
            label = f"{label} ({Path(path).name})"
        freshness = mapping(manifest.get("freshness"))
        warnings = sequence(manifest.get("warnings"))
        tidy_csv = manifest.get("tidy_csv")
        inputs = [
            item
            for item in sequence(manifest.get("inputs"))
            if isinstance(item, Mapping) and (item.get("path") or item.get("role"))
        ]
        summary = str(
            manifest.get("caption")
            or manifest.get("review_purpose")
            or mapping(manifest.get("metadata")).get("summary")
            or ""
        )
        choices.append(
            {
                "label": label,
                "title": title,
                "name": name,
                "kind": entry.get("kind") or manifest.get("kind") or "unknown",
                "path": path,
                "workdir": workdir,
                "path_label": compact_path(path, base=workdir),
                "filename": Path(path).name,
                "tidy_label": compact_path(tidy_csv, base=workdir) if tidy_csv else "none",
                "source_labels": [
                    f"{item.get('role') or 'input'}={compact_path(item.get('path'), base=workdir)}"
                    for item in inputs[:5]
                ],
                "freshness": freshness.get("status") or manifest.get("stale_state") or "unknown",
                "status": manifest.get("status"),
                "run_id": manifest.get("run_id"),
                "rounds": manifest.get("rounds"),
                "warning_count": len(warnings),
                "caption": summary,
                "alt_text": _plot_alt_text(
                    title=title,
                    kind=entry.get("kind") or manifest.get("kind"),
                    summary=summary,
                ),
                "entry": dict(entry),
                "manifest": dict(manifest),
            }
        )
    return {
        "plots_dir": plots_dir,
        "choices": choices,
        "missing_outputs": missing_outputs,
        "stale_artifacts": list(sequence(view_model.get("stale_artifacts"))),
    }


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
        {"field": "generated", "value": manifest.get("generated_at")},
        {"field": "run", "value": manifest.get("run_id") or "all runs"},
        {"field": "rounds", "value": manifest.get("rounds")},
        {"field": "media", "value": choice.get("path_label") or compact_path(choice.get("path"), base=base)},
        {"field": "tidy data", "value": choice.get("tidy_label") or compact_path(manifest.get("tidy_csv"), base=base)},
        {
            "field": "source data",
            "value": "; ".join(
                f"{item.get('role') or 'input'}={compact_path(item.get('path'), base=base)}" for item in inputs[:5]
            )
            or "not recorded",
        },
        {"field": "warnings", "value": len(sequence(manifest.get("warnings")))},
    ]


def build_notebook_plot_method_rows(choice: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build plot interpretation and math/data-contract rows."""

    manifest = mapping(choice.get("manifest"))
    metadata = mapping(manifest.get("metadata"))
    kind = str(choice.get("kind") or manifest.get("kind") or "unknown")
    return [
        {
            "section": "reading",
            "detail": str(choice.get("caption") or metadata.get("summary") or "No plot description recorded."),
        },
        {"section": "data shape", "detail": str(metadata.get("data_shape") or "not recorded")},
        {"section": "math", "detail": _plot_math_description(kind)},
        {"section": "parameters", "detail": _compact_params(manifest.get("params"))},
        {"section": "tidy schema", "detail": join_list(metadata.get("tidy_schema"), sep=", ")},
        {"section": "failure modes", "detail": join_list(metadata.get("failure_modes"), sep="; ")},
    ]


def _plot_alt_text(*, title: str, kind: Any, summary: str) -> str:
    kind_text = str(kind or "plot").replace("_", " ")
    summary_text = str(summary or "").strip()
    if summary_text:
        return f"{title}. {summary_text}"
    return f"{title}. OPAL {kind_text} visual for the selected campaign."


def _plot_math_description(kind: str) -> str:
    descriptions = {
        "metric_over_rounds": (
            "For each round and cohort, OPAL filters prediction rows, extracts the configured numeric metric, "
            "then computes requested summaries. mean = sum(x) / n; quantile summaries use the requested order "
            "statistic; count = n."
        ),
        "feature_importance_heatmap": (
            "OPAL builds a feature-by-round matrix from model feature_importance.csv artifacts. Each cell is the "
            "model-reported importance for one feature in one round; top_n keeps features with the largest maximum "
            "importance across rounds."
        ),
        "vector_summary_heatmap": (
            "For each round, cohort, and vector channel, OPAL aggregates the configured vector prediction field. "
            "The current primitive computes the mean channel value across matching prediction rows."
        ),
        "scatter_score_vs_rank": (
            "OPAL plots selected prediction rows by rank and score. Rank is selection order; score is the configured "
            "selection score or objective channel."
        ),
    }
    return descriptions.get(
        kind,
        "See the plot kind metadata for the exact data shape, required fields, parameters, and failure modes.",
    )


def _compact_params(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return "not recorded"
    return "; ".join(f"{key}={item}" for key, item in value.items())
