from __future__ import annotations

from typing import Any, Iterable, Mapping

from ._support import display_name, mapping
from .campaign_set_relationships import campaign_pair_contexts, metadata_fields, relationship_pair_membership
from .campaign_set_sources import (
    campaign_plot_manifest,
    finite_number,
    manifest_tidy_csv_path,
    read_csv_dict_rows,
)


def build_notebook_campaign_set_vector_reference_mse_rows(
    campaigns: Iterable[Mapping[str, Any]],
    *,
    plot_name: str,
    group_key: str,
    relationship: Mapping[str, Any] | None = None,
    cohort: str = "selected",
) -> list[dict[str, Any]]:
    """Read reference-MSE rows from vector_summary_heatmap tidy CSVs."""

    rows: list[dict[str, Any]] = []
    pair_membership = relationship_pair_membership(relationship)
    for campaign_model in campaigns:
        campaign = mapping(campaign_model.get("campaign"))
        slug = str(campaign.get("slug") or "unknown")
        pair_contexts = campaign_pair_contexts(campaign_model, pair_membership) if pair_membership else [None]
        if not pair_contexts:
            continue
        metadata = mapping(campaign.get("metadata"))
        manifest = campaign_plot_manifest(campaign_model, name=plot_name, kind="vector_summary_heatmap")
        if manifest is None:
            continue
        params = mapping(manifest.get("params"))
        tidy_path = manifest_tidy_csv_path(manifest)
        if tidy_path is None or not tidy_path.exists():
            continue
        group_value = str(metadata.get(str(group_key), "not recorded"))
        for raw in read_csv_dict_rows(tidy_path):
            if str(raw.get("row_type") or "") != "reference_mse":
                continue
            if str(raw.get("cohort") or "") != str(cohort):
                continue
            round_value = finite_number(raw.get("round"))
            metric_value = finite_number(raw.get("value"))
            if round_value is None or metric_value is None:
                continue
            for pair_context in pair_contexts:
                rows.append(
                    {
                        **metadata_fields(metadata),
                        **(pair_context or {}),
                        "round": int(round_value),
                        "cohort": str(raw.get("cohort") or cohort),
                        "metric": "reference_mse",
                        "summary": "mean",
                        "value": float(metric_value),
                        "cohort_count": finite_number(raw.get("n")),
                        "campaign": slug,
                        "campaign_label": display_name(slug),
                        "group_key": group_key,
                        "group": group_value,
                        "tidy_csv": str(tidy_path),
                        "metric_label": str(
                            params.get("reference_mse_metric_label")
                            or "MSE = mean((mean selected y_hat - reference)^2)"
                        ),
                        "legend_metric_label": str(params.get("reference_mse_legend_label") or "reference MSE"),
                        "metric_expression": str(
                            params.get("reference_mse_expression")
                            or "MSE = mean((mean selected y_hat - reference)^2); lower is better"
                        ),
                        **_reference_mse_axis_fields(params),
                    }
                )
    return rows


def _reference_mse_axis_fields(params: Mapping[str, Any]) -> dict[str, Any]:
    low, high = _axis_limits(params.get("reference_mse_y_limits", params.get("reference_mse_limits")))
    first_reference = _first_reference_line(params.get("reference_mse_reference_lines"))
    return {
        "axis_scale_class": str(params.get("reference_mse_scale_class") or "reference_mse").strip(),
        "y_axis_min": low,
        "y_axis_max": high,
        "y_axis_reference_value": first_reference.get("value"),
        "y_axis_reference_label": first_reference.get("label", ""),
        "y_axis_include_zero_tick": bool(params.get("reference_mse_include_zero_tick", True)),
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
    reference_value = finite_number(first.get("value"))
    if reference_value is None:
        return {}
    return {
        "value": reference_value,
        "label": str(first.get("label") or ""),
    }
