from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from ...plots._mpl_utils import pretty_label
from ._support import display_name, mapping, sequence
from .plot_scopes import sort_plot_scope_manifests


def build_notebook_campaign_set_group_options(campaigns: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return useful campaign metadata fields for grouping campaign-set comparisons."""

    campaign_list = [campaign_model for campaign_model in campaigns if isinstance(campaign_model, Mapping)]
    if len(campaign_list) <= 1:
        return []

    values_by_key: dict[str, list[str]] = {}
    slugs: list[str] = []
    for campaign_model in campaign_list:
        campaign = mapping(campaign_model.get("campaign"))
        slugs.append(str(campaign.get("slug") or "campaign"))
        metadata = mapping(campaign.get("metadata"))
        for key, value in metadata.items():
            if _is_groupable_metadata_value(value):
                values_by_key.setdefault(str(key), []).append(str(value))
    candidates = [
        key for key, values in values_by_key.items() if len(values) == len(campaign_list) and len(set(values)) > 1
    ]
    if slugs and len(set(slugs)) > 1:
        values_by_key["campaign"] = slugs
        candidates.append("campaign")
    if not candidates:
        return []

    def _sort_key(key: str) -> tuple[int, int, int, str]:
        lower = key.lower()
        semantic_rank = (
            0
            if any(token in lower for token in ("oracle_kind", "scenario", "condition", "group", "kind", "class"))
            else 1
        )
        identifier_penalty = 1 if any(token in lower for token in ("id", "split", "run", "hash", "path")) else 0
        campaign_penalty = 2 if lower == "campaign" else 0
        unique_count = len(set(values_by_key[key]))
        return (semantic_rank + identifier_penalty + campaign_penalty, -unique_count, len(lower), lower)

    selected: list[str] = []
    seen_partitions: set[tuple[int, ...]] = set()
    for key in sorted(candidates, key=_sort_key):
        signature = _partition_signature(values_by_key[key])
        if signature in seen_partitions:
            continue
        seen_partitions.add(signature)
        selected.append(key)
    return selected


def build_notebook_campaign_set_metric_comparison_rows(
    campaigns: Iterable[Mapping[str, Any]],
    *,
    plot_name: str,
    group_key: str | None = None,
    summary: str = "median",
) -> list[dict[str, Any]]:
    """Read compatible metric_over_rounds tidy CSVs across a campaign set."""

    if not str(plot_name or "").strip():
        return []
    rows: list[dict[str, Any]] = []
    for campaign_model in campaigns:
        campaign = mapping(campaign_model.get("campaign"))
        slug = str(campaign.get("slug") or "unknown")
        campaign_label = display_name(slug)
        metadata = mapping(campaign.get("metadata"))
        group_value = slug if group_key in (None, "", "campaign") else str(metadata.get(str(group_key), "not recorded"))
        manifest = _campaign_plot_manifest(campaign_model, name=plot_name, kind="metric_over_rounds")
        if manifest is None:
            continue
        tidy_path = _manifest_tidy_csv_path(manifest)
        if tidy_path is None or not tidy_path.exists():
            continue
        for row in _read_metric_tidy_rows(tidy_path, summary=summary):
            rows.append(
                {
                    **row,
                    "campaign": slug,
                    "campaign_label": campaign_label,
                    "group_key": group_key or "campaign",
                    "group": group_value,
                    "tidy_csv": str(tidy_path),
                }
            )
    return rows


def render_notebook_campaign_set_metric_comparison_image(
    rows: Iterable[Mapping[str, Any]],
    *,
    title: str,
    group_key: str,
    dpi: int = 180,
) -> dict[str, Any] | None:
    """Render a compact PNG comparison from campaign-set metric rows."""

    data = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and _finite_number(row.get("value")) is not None
        and _finite_number(row.get("round")) is not None
    ]
    if not data:
        return None

    import matplotlib.pyplot as plt
    import numpy as np

    from ...plots._mpl_utils import (
        DEFAULT_SQUARE_FIGSIZE,
        apply_notebook_axes_style,
        apply_plot_style,
        categorical_color,
        categorical_linestyle,
        categorical_marker,
        legend_below_single_row,
        pretty_title,
    )

    grouped: dict[str, dict[int, list[float]]] = {}
    counts_by_group_round: dict[tuple[str, int], list[float]] = {}
    metric = str(data[0].get("metric") or "value")
    cohort = str(data[0].get("cohort") or "")
    summary = str(data[0].get("summary") or "median")
    pairs = {(str(row.get("metric") or ""), str(row.get("cohort") or "")) for row in data}
    if len(pairs) != 1:
        rendered = ", ".join(f"{metric_name}/{cohort_name}" for metric_name, cohort_name in sorted(pairs))
        raise ValueError(f"Campaign-set metric comparison requires exactly one metric/cohort pair; got {rendered}.")
    for row in data:
        group = str(row.get("group") or "not recorded")
        round_index = int(_finite_number(row.get("round")))
        value = float(_finite_number(row.get("value")))
        grouped.setdefault(group, {}).setdefault(round_index, []).append(value)
        count_value = _finite_number(row.get("cohort_count"))
        if count_value is not None:
            counts_by_group_round.setdefault((group, round_index), []).append(float(count_value))

    apply_plot_style()
    fig, ax = plt.subplots(figsize=DEFAULT_SQUARE_FIGSIZE)
    apply_notebook_axes_style(ax, square=True)
    group_labels = sorted(grouped)
    for index, group in enumerate(group_labels):
        by_round = grouped[group]
        xs = sorted(by_round)
        ys = [float(np.median(by_round[round_index])) for round_index in xs]
        ax.plot(
            xs,
            ys,
            color=categorical_color(index),
            marker=categorical_marker(index),
            linestyle=categorical_linestyle(index),
            linewidth=2.4,
            markersize=7,
            label=pretty_label(group),
        )

    ax.set_xlabel("Round")
    ax.set_ylabel(f"{pretty_label(summary)} {pretty_label(metric)}")
    rendered_title = pretty_title(title)
    if cohort and str(cohort).lower() != "selected" and pretty_label(cohort).lower() not in rendered_title.lower():
        rendered_title = f"{rendered_title} ({pretty_label(cohort)})"
    count_text = _campaign_set_count_text(counts_by_group_round, cohort=cohort)
    if count_text:
        rendered_title = f"{rendered_title}\n{count_text}"
    ax.set_title(rendered_title)
    ax.set_xticks(sorted({int(_finite_number(row.get("round"))) for row in data}))
    used_bottom_legend = legend_below_single_row(fig, ax, bottom=0.10)
    if not used_bottom_legend:
        fig.tight_layout(pad=0.35)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=int(dpi), facecolor="white")
    plt.close(fig)
    group_text = ", ".join(pretty_label(group) for group in group_labels)
    return {
        "image_bytes": buffer.getvalue(),
        "alt_text": (
            f"Campaign-set comparison for {rendered_title}. X axis is OPAL round; "
            f"Y axis is {summary} {metric}; color, marker, and line style encode {group_key}. "
            f"Groups shown: {group_text}."
        ),
        "caption": (
            f"Campaign-set comparison grouped by `{group_key}` for `{metric}` / `{cohort}`. "
            "Values are median across campaigns per group/round."
        ),
        "group_count": len(group_labels),
        "row_count": len(data),
    }


def _is_groupable_metadata_value(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) and str(value).strip() != ""


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
        round_value = _finite_number(raw.get("round"))
        count_value = _finite_number(raw.get("value"))
        if round_value is None or count_value is None:
            continue
        counts[(int(round_value), str(raw.get("cohort") or ""), str(raw.get("metric") or ""))] = float(count_value)
    for raw in raw_rows:
        if str(raw.get("summary") or "") != str(summary):
            continue
        round_value = _finite_number(raw.get("round"))
        metric_value = _finite_number(raw.get("value"))
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


def _partition_signature(values: Iterable[str]) -> tuple[int, ...]:
    seen: dict[str, int] = {}
    signature: list[int] = []
    for value in values:
        token = str(value)
        if token not in seen:
            seen[token] = len(seen)
        signature.append(seen[token])
    return tuple(signature)


def _campaign_set_count_text(counts_by_group_round: Mapping[tuple[str, int], list[float]], *, cohort: str) -> str:
    if not counts_by_group_round:
        return ""
    values = [
        int(round(float(value)))
        for values in counts_by_group_round.values()
        for value in values
        if _finite_number(value) is not None
    ]
    if not values:
        return ""
    unique = sorted(set(values))
    cohort_text = pretty_label(cohort or "cohort")
    if len(unique) == 1:
        return f"{cohort_text} n={unique[0]} per campaign-round"
    return f"{cohort_text} n={min(unique)}-{max(unique)} per campaign-round"


def _finite_number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
