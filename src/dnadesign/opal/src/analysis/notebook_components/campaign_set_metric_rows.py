from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from ._support import display_name, mapping, sequence
from .campaign_set_relationships import campaign_pair_contexts, metadata_fields, relationship_pair_membership
from .plot_scopes import sort_plot_scope_manifests


def build_notebook_campaign_set_metric_comparison_rows(
    campaigns: Iterable[Mapping[str, Any]],
    *,
    plot_name: str,
    group_key: str | None = None,
    summary: str = "median",
    relationship: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Read compatible metric_over_rounds tidy CSVs across a campaign set."""

    if not str(plot_name or "").strip():
        return []
    rows: list[dict[str, Any]] = []
    pair_membership = relationship_pair_membership(relationship)
    for campaign_model in campaigns:
        campaign = mapping(campaign_model.get("campaign"))
        slug = str(campaign.get("slug") or "unknown")
        pair_contexts = campaign_pair_contexts(campaign_model, pair_membership) if pair_membership else [None]
        if not pair_contexts:
            continue
        campaign_label = display_name(slug)
        metadata = mapping(campaign.get("metadata"))
        group_value = slug if group_key in (None, "", "campaign") else str(metadata.get(str(group_key), "not recorded"))
        manifest = _campaign_plot_manifest(campaign_model, name=plot_name, kind="metric_over_rounds")
        if manifest is None:
            continue
        params = mapping(manifest.get("params"))
        tidy_path = _manifest_tidy_csv_path(manifest)
        if tidy_path is None or not tidy_path.exists():
            continue
        for row in _read_metric_tidy_rows(tidy_path, summary=summary):
            for pair_context in pair_contexts:
                rows.append(
                    {
                        **row,
                        **metadata_fields(metadata),
                        **(pair_context or {}),
                        "campaign": slug,
                        "campaign_label": campaign_label,
                        "group_key": group_key or "campaign",
                        "group": group_value,
                        "tidy_csv": str(tidy_path),
                        "metric_label": _manifest_param_text(params, "metric_label", "score_label", "y_label"),
                        "legend_metric_label": _manifest_param_text(
                            params,
                            "legend_metric_label",
                            "metric_short_label",
                            "score_short_label",
                        ),
                        "metric_expression": _manifest_param_text(
                            params,
                            "metric_expression",
                            "score_expression",
                            "loss_expression",
                        ),
                        "collection_visual_label": _manifest_param_text(params, "collection_visual_label"),
                        **_manifest_axis_fields(params),
                    }
                )
    return rows


def finite_number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _campaign_plot_manifest(
    campaign_model: Mapping[str, Any],
    *,
    name: str,
    kind: str,
) -> Mapping[str, Any] | None:
    candidates = [
        manifest
        for manifest in sequence(campaign_model.get("plot_manifests"))
        if isinstance(manifest, Mapping)
        and manifest.get("status") == "written"
        and str(manifest.get("name") or "") == str(name)
        and str(manifest.get("kind") or "") == str(kind)
    ]
    if not candidates:
        return None
    return sort_plot_scope_manifests(candidates)[0]


def _manifest_tidy_csv_path(manifest: Mapping[str, Any]) -> Path | None:
    tidy_csv = manifest.get("tidy_csv")
    if tidy_csv not in (None, ""):
        return Path(str(tidy_csv))
    for output in sequence(manifest.get("outputs")):
        if isinstance(output, Mapping) and output.get("role") == "tidy_csv" and output.get("path"):
            return Path(str(output["path"]))
    return None


def _read_metric_tidy_rows(path: Path, *, summary: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            raw_rows.append(dict(raw))
    counts: dict[tuple[int, str, str], float] = {}
    for raw in raw_rows:
        if str(raw.get("summary") or "") != "count":
            continue
        round_value = finite_number(raw.get("round"))
        count_value = finite_number(raw.get("value"))
        if round_value is None or count_value is None:
            continue
        counts[(int(round_value), str(raw.get("cohort") or ""), str(raw.get("metric") or ""))] = float(count_value)
    for raw in raw_rows:
        if str(raw.get("summary") or "") != str(summary):
            continue
        round_value = finite_number(raw.get("round"))
        metric_value = finite_number(raw.get("value"))
        if round_value is None or metric_value is None:
            continue
        cohort = raw.get("cohort") or ""
        metric = raw.get("metric") or ""
        rows.append(
            {
                "round": int(round_value),
                "cohort": cohort,
                "metric": metric,
                "summary": raw.get("summary") or summary,
                "value": float(metric_value),
                "cohort_count": counts.get((int(round_value), str(cohort), str(metric))),
            }
        )
    return rows


def _manifest_param_text(params: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = params.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _manifest_axis_fields(params: Mapping[str, Any]) -> dict[str, Any]:
    y_axis = params.get("y_axis") if isinstance(params.get("y_axis"), Mapping) else {}
    limits = y_axis.get("limits", params.get("y_limits"))
    reference_lines = y_axis.get("reference_lines", params.get("y_reference_lines"))
    first_reference = _first_reference_line(reference_lines)
    low, high = _axis_limits(limits)
    return {
        "axis_scale_class": str(y_axis.get("scale_class") or params.get("y_scale_class") or "").strip(),
        "y_axis_min": low,
        "y_axis_max": high,
        "y_axis_reference_value": first_reference.get("value"),
        "y_axis_reference_label": first_reference.get("label", ""),
        "y_axis_include_zero_tick": bool(y_axis.get("include_zero_tick", params.get("include_zero_tick", False))),
    }


def _axis_limits(value: Any) -> tuple[float | None, float | None]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None, None
    return finite_number(value[0]), finite_number(value[1])


def _first_reference_line(value: Any) -> dict[str, Any]:
    if not isinstance(value, list) or not value:
        return {}
    first = value[0]
    if not isinstance(first, Mapping):
        return {}
    return {
        "value": finite_number(first.get("value")),
        "label": str(first.get("label") or ""),
    }
