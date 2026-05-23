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
                    params=manifest.get("params"),
                    metadata=manifest.get("metadata"),
                    rounds=manifest.get("rounds"),
                    run_id=manifest.get("run_id"),
                    freshness=freshness.get("status") or manifest.get("stale_state"),
                    warning_count=len(warnings),
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


def build_notebook_plot_method_sections(choice: Mapping[str, Any]) -> dict[str, str]:
    """Build readable accordion sections for the selected plot's method."""

    rows = {str(row["section"]): str(row["detail"]) for row in build_notebook_plot_method_rows(choice)}
    title = str(choice.get("title") or display_name(choice.get("name"))).strip()
    kind = str(choice.get("kind") or "unknown").replace("_", " ")
    rounds = _rounds_text(choice.get("rounds"))
    freshness = str(choice.get("freshness") or "unknown")
    warnings = int(choice.get("warning_count") or 0)
    return {
        "Read": (f"{title} shows a {kind} view for {rounds}. {rows.get('reading', 'No plot description recorded.')}"),
        "Math": rows.get("math", "No math description recorded."),
        "Data contract": (
            f"Data shape: {rows.get('data shape', 'not recorded')}.\n\n"
            f"Parameters: {rows.get('parameters', 'not recorded')}.\n\n"
            f"Tidy schema: {rows.get('tidy schema', 'not recorded')}.\n\n"
            f"Failure modes: {rows.get('failure modes', 'not recorded')}.\n\n"
            f"Freshness: `{freshness}`. Warnings: `{warnings}`."
        ),
    }


def _plot_alt_text(
    *,
    title: str,
    kind: Any,
    summary: str,
    params: Any,
    metadata: Any,
    rounds: Any,
    run_id: Any,
    freshness: Any,
    warning_count: int,
) -> str:
    kind_text = str(kind or "plot").replace("_", " ")
    summary_text = str(summary or "").strip()
    scope = _rounds_text(rounds)
    run_text = "all runs" if run_id in (None, "") else f"run {run_id}"
    field_text = _plot_field_text(kind=str(kind or ""), params=mapping(params), metadata=mapping(metadata))
    quality = f"freshness {freshness or 'unknown'}"
    if int(warning_count) > 0:
        quality += f", {int(warning_count)} warnings"
    if summary_text:
        return f"{title}. {summary_text} {field_text} Scope: {scope}, {run_text}; {quality}."
    return f"{title}. OPAL {kind_text} visual. {field_text} Scope: {scope}, {run_text}; {quality}."


def _rounds_text(value: Any) -> str:
    if value in (None, ""):
        return "the selected round"
    if value == "all":
        return "all rounds"
    items = sequence(value)
    if len(items) == 1:
        return f"round {items[0]}"
    return "rounds " + ", ".join(str(item) for item in items)


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
        "percent_high_activity_over_rounds": (
            "For each round, OPAL counts rows where score >= threshold. percent_high = 100 * high / total; "
            "violin and swarm layers show the underlying score distribution when enabled."
        ),
        "feature_importance_bars": (
            "OPAL reads per-round model feature_importance.csv artifacts. Bars encode model-reported importance, "
            "with ordering controlled by the configured order policy."
        ),
        "fold_change_vs_logic_fidelity": (
            "For SFXI, logic fidelity is clipped 1 - ||v - p||2 / D, where v is the predicted logic vector, "
            "p is the setpoint, and D is the worst-corner distance. The y-axis is the configured effect or score field."
        ),
        "sfxi_logic_fidelity_closeness": (
            "For each observed label, OPAL uses the first four SFXI components as logic values and computes "
            "mean squared error against the setpoint: MSE = mean((v - p)^2). Lower MSE means closer logic behavior."
        ),
        "sfxi_factorial_effects": (
            "With state order 00,10,01,11, A effect = ((v10 + v11) - (v00 + v01)) / 2, "
            "B effect = ((v01 + v11) - (v00 + v10)) / 2, and interaction = ((v11 + v00) - (v10 + v01)) / 2."
        ),
        "sfxi_setpoint_sweep": (
            "OPAL evaluates label vectors against a setpoint library. Logic fidelity is 1 - ||v - p||2 / D, "
            "effect is scaled by a percentile denominator, and score = logic_fidelity^beta * effect_scaled^gamma."
        ),
        "sfxi_support_diagnostics": (
            "For each candidate, OPAL computes Euclidean distance in four-channel logic space to the nearest "
            "labeled point. Larger distances flag extrapolation risk."
        ),
        "sfxi_uncertainty": (
            "For models with ensemble predictions, OPAL computes the standard deviation of objective scores "
            "across estimators after inverse y-ops and SFXI scoring."
        ),
        "sfxi_intensity_scaling": (
            "OPAL recovers linear intensity as max(0, 2^y_star - delta), computes weighted raw effect, "
            "uses a configured percentile denominator, then clips effect_scaled = effect_raw / denom to [0, 1]."
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


def _plot_field_text(*, kind: str, params: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    fields: list[str] = []
    for key in (
        "metric",
        "summary",
        "summaries",
        "cohort",
        "score_field",
        "rank_mode",
        "threshold",
        "y_axis",
        "hue",
        "hue_field",
        "size_by",
        "vector_field",
        "aggregation",
    ):
        value = params.get(key)
        if value not in (None, "", []):
            fields.append(f"{key}={value}")
    shape = metadata.get("data_shape")
    if shape not in (None, ""):
        fields.append(f"data_shape={shape}")
    if not fields:
        return f"Plot kind: {str(kind or 'unknown').replace('_', ' ')}."
    return "Encoded fields: " + "; ".join(str(item) for item in fields[:8]) + "."
