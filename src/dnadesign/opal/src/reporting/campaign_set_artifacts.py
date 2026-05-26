"""Materialized campaign-set visual artifacts."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..analysis.campaign_set import build_campaign_set_collection_visual_model
from ..analysis.notebook_components import (
    build_notebook_campaign_set_metric_comparison_rows,
    build_notebook_campaign_set_plot_gallery_items,
    build_notebook_campaign_set_vector_heatmap_rows,
    build_notebook_campaign_set_vector_reference_mse_rows,
    render_notebook_campaign_set_metric_comparison_image,
    render_notebook_campaign_set_plot_gallery_image,
    render_notebook_campaign_set_vector_heatmap_comparison_image,
)
from ..core.utils import ExitCodes, OpalError, now_iso, read_json, write_json

COLLECTION_VISUAL_ARTIFACT_SCHEMA_VERSION = "opal.collection_visual_artifact.v1"
COLLECTION_VISUAL_MANIFEST_INDEX_SCHEMA_VERSION = "opal.collection_visual_manifest_index.v1"


def materialize_campaign_set_collection_visuals(
    campaigns: Iterable[Mapping[str, Any]],
    *,
    collection: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write collection-level comparison visuals and their manifest index."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _clear_owned_collection_visual_outputs(output_path)
    campaign_list = [campaign for campaign in campaigns if isinstance(campaign, Mapping)]
    model = build_campaign_set_collection_visual_model(campaign_list, collection)
    visuals = [
        _materialize_visual(campaign_list, visual=visual, collection=collection, output_dir=output_path)
        for visual in model["visuals"]
    ]
    index = {
        "schema_version": COLLECTION_VISUAL_MANIFEST_INDEX_SCHEMA_VERSION,
        "generated_at": now_iso(),
        "collection_id": collection.get("collection_id"),
        "output_dir": str(output_path),
        "comparison_set_count": model["comparison_set_count"],
        "comparison_sets": model["comparison_sets"],
        "visual_count": len(visuals),
        "visuals": visuals,
    }
    write_json(output_path / "collection_visual_manifest.json", index)
    return index


def _clear_owned_collection_visual_outputs(output_dir: Path) -> None:
    """Remove stale materialized visual files before writing a fresh index."""

    for child in output_dir.iterdir():
        if not child.is_file():
            continue
        if child.name == "collection_visual_manifest.json" or child.name.endswith((".manifest.json", ".csv", ".png")):
            child.unlink()


def load_collection_visual_manifest_index(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise OpalError(
            f"Collection visual manifest index is not a JSON object: {path}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if payload.get("schema_version") != COLLECTION_VISUAL_MANIFEST_INDEX_SCHEMA_VERSION:
        raise OpalError(
            f"Unsupported collection visual manifest index schema: {payload.get('schema_version')!r}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return payload


def _materialize_visual(
    campaigns: list[Mapping[str, Any]],
    *,
    visual: Mapping[str, Any],
    collection: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    visual_id = str(visual.get("id") or "collection_visual")
    set_key = str(visual.get("comparison_set_key") or "")
    stem = _artifact_stem("__".join(part for part in [set_key, visual_id] if part))
    tidy_path = output_dir / f"{stem}.csv"
    media_path = output_dir / f"{stem}.png"
    manifest_path = output_dir / f"{stem}.manifest.json"
    rows, rendered, input_paths = _render_visual(
        campaigns,
        visual=visual,
        media_path=media_path,
    )
    if not rows or rendered is None:
        raise OpalError(
            f"Campaign-set comparison view {visual_id!r} has no matching source rows.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    _write_rows_csv(rows, tidy_path)
    media_path.write_bytes(rendered["image_bytes"])
    inputs = [_file_entry(path, role="source") for path in input_paths]
    outputs = [_file_entry(media_path, role="media"), _file_entry(tidy_path, role="tidy_csv")]
    manifest = {
        "schema_version": COLLECTION_VISUAL_ARTIFACT_SCHEMA_VERSION,
        "generated_at": now_iso(),
        "collection_id": collection.get("collection_id"),
        "visual_id": visual_id,
        "label": rendered.get("label") or visual.get("label"),
        "title": rendered.get("title") or visual.get("title"),
        "surface_kind": visual.get("surface_kind"),
        "kind": visual.get("kind"),
        "view_kind": visual.get("view_kind"),
        "source_plot_name": visual.get("source_plot_name"),
        "source_plot_kind": visual.get("source_plot_kind"),
        "comparison_scope": visual.get("comparison_scope"),
        "comparison_set_key": visual.get("comparison_set_key"),
        "comparison_set_label": visual.get("comparison_set_label"),
        "comparison_set_match": visual.get("comparison_set_match"),
        "comparison_replicate_count": visual.get("comparison_replicate_count"),
        "match_filters": visual.get("match_filters"),
        "relationship_id": visual.get("relationship_id"),
        "relationship_kind": visual.get("relationship_kind"),
        "group_key": visual.get("group_key"),
        "metric": visual.get("metric"),
        "cohort": visual.get("cohort"),
        "summary": visual.get("summary"),
        "interval_kind": visual.get("interval_kind"),
        "confidence_level": visual.get("confidence_level"),
        "interpretation_note": visual.get("interpretation_note"),
        "metric_label": rendered.get("metric_label") or visual.get("metric_label"),
        "legend_metric_label": rendered.get("legend_metric_label") or visual.get("legend_metric_label"),
        "metric_expression": rendered.get("metric_expression") or visual.get("metric_expression"),
        "plot_question": rendered.get("plot_question"),
        "target_vector_label": rendered.get("target_vector_label"),
        "mse_formula": rendered.get("mse_formula"),
        "visual_contract": rendered.get("visual_contract"),
        "axis_scale": rendered.get("axis_scale"),
        "comparison_unit": _comparison_unit(visual),
        "row_count": len(rows),
        "group_count": rendered.get("group_count"),
        "inputs": inputs,
        "outputs": outputs,
        "tidy_csv": str(tidy_path),
        "path": str(media_path),
        "manifest_path": str(manifest_path),
        "freshness": _freshness_entry(inputs=inputs, outputs=outputs),
        "interval": rendered.get("interval"),
        "caption": rendered.get("caption"),
        "alt_text": rendered.get("alt_text"),
    }
    write_json(manifest_path, manifest)
    return manifest


def _render_visual(
    campaigns: list[Mapping[str, Any]],
    *,
    visual: Mapping[str, Any],
    media_path: Path,
) -> tuple[list[Mapping[str, Any]], dict[str, Any] | None, list[Path]]:
    surface_kind = str(visual.get("surface_kind") or "")
    if surface_kind == "campaign_set_metric_comparison":
        rows = build_notebook_campaign_set_metric_comparison_rows(
            campaigns,
            plot_name=str(visual["source_plot_name"]),
            group_key=str(visual["group_key"]),
            summary=str(visual["summary"]),
            relationship=visual,
        )
        rows = _filter_metric_rows(rows, visual=visual)
        rendered = render_notebook_campaign_set_metric_comparison_image(
            rows,
            title=str(visual.get("title") or visual.get("label") or visual.get("id")),
            group_key=str(visual["group_key"]),
            interval_kind=str(visual.get("interval_kind") or "none"),
            confidence_level=visual.get("confidence_level") if visual.get("confidence_level") is not None else None,
            interpretation_note=str(visual.get("interpretation_note") or ""),
        )
        return rows, rendered, _source_tidy_paths(rows)
    if surface_kind == "campaign_set_vector_reference_mse_comparison":
        rows = build_notebook_campaign_set_vector_reference_mse_rows(
            campaigns,
            plot_name=str(visual["source_plot_name"]),
            group_key=str(visual["group_key"]),
            relationship=visual,
            cohort=str(visual.get("cohort") or "selected"),
        )
        rows = _filter_metric_rows(rows, visual=visual)
        rendered = render_notebook_campaign_set_metric_comparison_image(
            rows,
            title=str(visual.get("title") or visual.get("label") or visual.get("id")),
            group_key=str(visual["group_key"]),
            interval_kind=str(visual.get("interval_kind") or "none"),
            confidence_level=visual.get("confidence_level") if visual.get("confidence_level") is not None else None,
            interpretation_note=str(visual.get("interpretation_note") or ""),
        )
        return rows, rendered, _source_tidy_paths(rows)
    if surface_kind == "campaign_set_plot_gallery":
        items = build_notebook_campaign_set_plot_gallery_items(
            campaigns,
            plot_name=str(visual["source_plot_name"]),
            plot_kind=str(visual["source_plot_kind"]),
            group_key=str(visual["group_key"]),
            relationship=visual,
        )
        rendered = render_notebook_campaign_set_plot_gallery_image(
            items,
            title=str(visual.get("title") or visual.get("label") or visual.get("id")),
            group_key=str(visual["group_key"]),
        )
        return items, rendered, _source_media_paths(items)
    if surface_kind == "campaign_set_vector_heatmap_comparison":
        rows = build_notebook_campaign_set_vector_heatmap_rows(
            campaigns,
            plot_name=str(visual["source_plot_name"]),
            group_key=str(visual["group_key"]),
            relationship=visual,
            cohort=str(visual.get("cohort") or "selected"),
        )
        rendered = render_notebook_campaign_set_vector_heatmap_comparison_image(
            rows,
            title=str(visual.get("title") or visual.get("label") or visual.get("id")),
            group_key=str(visual["group_key"]),
            interval_kind=str(visual.get("interval_kind") or "none"),
            interpretation_note=str(visual.get("interpretation_note") or ""),
        )
        return rows, rendered, _source_tidy_paths(rows)
    raise OpalError(
        f"Unsupported collection visual surface_kind: {surface_kind!r}",
        ExitCodes.CONTRACT_VIOLATION,
    )


def _filter_metric_rows(rows: list[dict[str, Any]], *, visual: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("metric") or "") == str(visual["metric"])
        and str(row.get("cohort") or "") == str(visual["cohort"])
    ]


def _write_rows_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    columns = [
        "row_type",
        "round",
        "cohort",
        "metric",
        "summary",
        "channel",
        "value",
        "cohort_count",
        "campaign",
        "campaign_label",
        "group_key",
        "group",
        "relationship_kind",
        "role_dimension",
        "comparison_role",
        "pair_key",
        "match_key",
        "replicate_key",
        "replicate_on",
        "comparison_unit_key",
        "axis_scale_class",
        "y_axis_min",
        "y_axis_max",
        "y_axis_reference_value",
        "y_axis_reference_label",
        "y_axis_include_zero_tick",
        "media_path",
        "tidy_csv",
    ]
    extras = sorted({str(key) for row in rows for key in row if str(key).startswith("metadata__")})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*columns, *extras], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _source_tidy_paths(rows: Iterable[Mapping[str, Any]]) -> list[Path]:
    paths = sorted({str(row.get("tidy_csv") or "") for row in rows if row.get("tidy_csv")})
    return [Path(path) for path in paths]


def _source_media_paths(rows: Iterable[Mapping[str, Any]]) -> list[Path]:
    paths = sorted({str(row.get("media_path") or "") for row in rows if row.get("media_path")})
    return [Path(path) for path in paths]


def _artifact_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "collection_visual"


def _file_entry(path: str | Path, *, role: str) -> dict[str, Any]:
    file_path = Path(path)
    entry: dict[str, Any] = {"role": role, "path": str(file_path), "exists": file_path.exists()}
    if file_path.exists():
        stat = file_path.stat()
        entry.update({"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return entry


def _freshness_entry(*, inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    input_mtimes = [
        int(entry["mtime_ns"]) for entry in inputs if entry.get("exists") and isinstance(entry.get("mtime_ns"), int)
    ]
    output_mtimes = [
        int(entry["mtime_ns"]) for entry in outputs if entry.get("exists") and isinstance(entry.get("mtime_ns"), int)
    ]
    latest_input = max(input_mtimes) if input_mtimes else None
    oldest_output = min(output_mtimes) if output_mtimes else None
    if not output_mtimes:
        status = "missing_outputs"
    elif latest_input is not None and oldest_output is not None and oldest_output < latest_input:
        status = "stale"
    else:
        status = "fresh"
    return {
        "schema_version": "opal.collection_visual_freshness.v1",
        "status": status,
        "latest_input_mtime_ns": latest_input,
        "oldest_output_mtime_ns": oldest_output,
        "inputs": inputs,
    }


def _comparison_unit(visual: Mapping[str, Any]) -> str:
    return "relationship_pair" if visual.get("pairs") else "campaign"
