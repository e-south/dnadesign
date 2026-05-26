"""Campaign-collection manifest contracts for OPAL notebooks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from ..core.utils import ExitCodes, OpalError

CAMPAIGN_COLLECTION_SCHEMA_VERSION = "opal.campaign_collection.v2"
COMPARISON_VIEW_KINDS = {
    "metric_over_rounds_comparison",
    "paired_plot_gallery",
    "vector_heatmap_comparison",
    "vector_reference_mse_over_rounds_comparison",
}
INTERVAL_KINDS = {"none", "iqr", "student_t_mean_ci"}
COMPARISON_SCOPES = {"collection", "comparison_set"}


def load_campaign_collection_manifest(
    path: str | Path,
    campaigns: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Load and validate an optional campaign-set relationship manifest."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise OpalError(f"Campaign collection manifest not found: {manifest_path}", ExitCodes.BAD_ARGS)
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OpalError(f"Failed to read campaign collection manifest: {manifest_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise OpalError("Campaign collection manifest must be a mapping.", ExitCodes.CONTRACT_VIOLATION)
    schema = raw.get("schema_version")
    if schema != CAMPAIGN_COLLECTION_SCHEMA_VERSION:
        raise OpalError(f"Unsupported campaign collection schema_version: {schema!r}", ExitCodes.CONTRACT_VIOLATION)

    collection_id = _required_string(raw.get("collection_id"), field="collection_id")
    dimensions = _dimension_rows(raw.get("dimensions"))
    dimension_ids = [row["id"] for row in dimensions]
    campaign_rows = _campaign_rows(campaigns)
    relationships = [
        _relationship_payload(row, index=index, dimensions=dimension_ids, campaign_rows=campaign_rows)
        for index, row in enumerate(_mapping_list(raw.get("relationships"), field="relationships"))
    ]
    _require_unique_ids(relationships, field="relationships")
    relationships_by_id = {str(row["id"]): row for row in relationships}
    comparison_views = [
        _comparison_view_payload(
            row,
            index=index,
            dimensions=dimension_ids,
            relationships_by_id=relationships_by_id,
        )
        for index, row in enumerate(_mapping_list(raw.get("comparison_views"), field="comparison_views"))
    ]
    _require_unique_ids(comparison_views, field="comparison_views")
    return {
        "schema_version": CAMPAIGN_COLLECTION_SCHEMA_VERSION,
        "collection_id": collection_id,
        "path": str(manifest_path),
        "dimensions": dimensions,
        "dimension_ids": dimension_ids,
        "relationships": relationships,
        "relationship_count": len(relationships),
        "comparison_views": comparison_views,
        "comparison_view_count": len(comparison_views),
        "comparison_lenses": [_comparison_lens(row) for row in relationships],
    }


def _relationship_payload(
    row: Mapping[str, Any],
    *,
    index: int,
    dimensions: list[str],
    campaign_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    relationship_id = _required_string(row.get("id"), field=f"relationships[{index}].id")
    kind = _required_string(row.get("kind"), field=f"relationships[{index}].kind")
    left_role = _required_string(row.get("left_role"), field=f"relationships[{index}].left_role")
    right_role = _required_string(row.get("right_role"), field=f"relationships[{index}].right_role")
    match_on = _string_list(row.get("match_on"), field=f"relationships[{index}].match_on", allow_empty=False)
    _require_declared_dimensions(match_on, dimensions, field=f"relationships[{index}].match_on")
    replicate_on = _string_list(row.get("replicate_on"), field=f"relationships[{index}].replicate_on", allow_empty=True)
    _require_declared_dimensions(replicate_on, dimensions, field=f"relationships[{index}].replicate_on")
    if any(dimension not in match_on for dimension in replicate_on):
        raise OpalError(
            f"Campaign collection field relationships[{index}].replicate_on must be a subset of match_on.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    role_dimension = str(row.get("role_dimension") or "").strip()
    if role_dimension:
        _require_declared_dimensions([role_dimension], dimensions, field=f"relationships[{index}].role_dimension")
    else:
        role_dimension = _infer_role_dimension(
            dimensions=dimensions,
            match_on=match_on,
            left_role=left_role,
            right_role=right_role,
            campaign_rows=campaign_rows,
            index=index,
        )
    pairs = _relationship_pairs(
        campaign_rows=campaign_rows,
        role_dimension=role_dimension,
        left_role=left_role,
        right_role=right_role,
        match_on=match_on,
    )
    if not pairs:
        raise OpalError(
            f"Campaign collection relationship {relationship_id!r} matched no campaign pairs.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return {
        "id": relationship_id,
        "kind": kind,
        "label": str(row.get("label") or _relationship_label(kind=kind, role_dimension=role_dimension)),
        "role_dimension": role_dimension,
        "left_role": left_role,
        "right_role": right_role,
        "match_on": match_on,
        "replicate_on": replicate_on,
        "pair_count": len(pairs),
        "pairs": pairs,
    }


def _relationship_pairs(
    *,
    campaign_rows: list[dict[str, Any]],
    role_dimension: str,
    left_role: str,
    right_role: str,
    match_on: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in campaign_rows:
        metadata = row["metadata"]
        if role_dimension not in metadata or any(key not in metadata for key in match_on):
            continue
        grouped[tuple(str(metadata[key]) for key in match_on)].append(row)

    pairs: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        left = [row for row in rows if str(row["metadata"].get(role_dimension)) == left_role]
        right = [row for row in rows if str(row["metadata"].get(role_dimension)) == right_role]
        if len(left) > 1 or len(right) > 1:
            rendered = ", ".join(f"{name}={value}" for name, value in zip(match_on, key, strict=True))
            raise OpalError(
                "Campaign collection relationship is ambiguous for match "
                f"{rendered}: left={len(left)}, right={len(right)}.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        for left_row in left:
            for right_row in right:
                pair = {
                    "left": left_row["member_key"],
                    "right": right_row["member_key"],
                    "match": dict(zip(match_on, key, strict=True)),
                }
                if left_row["member_key"] != left_row["slug"]:
                    pair["left_slug"] = left_row["slug"]
                if right_row["member_key"] != right_row["slug"]:
                    pair["right_slug"] = right_row["slug"]
                pairs.append(pair)
    return pairs


def _comparison_lens(relationship: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": relationship["id"],
        "kind": relationship["kind"],
        "label": _relationship_label(
            kind=str(relationship["kind"]),
            role_dimension=str(relationship["role_dimension"]),
        ),
        "group_key": relationship["role_dimension"],
        "role_dimension": relationship["role_dimension"],
        "left_role": relationship["left_role"],
        "right_role": relationship["right_role"],
        "match_on": list(relationship["match_on"]),
        "replicate_on": list(relationship.get("replicate_on") or []),
        "pair_count": int(relationship["pair_count"]),
        "pairs": list(relationship.get("pairs") or []),
    }


def _comparison_view_payload(
    row: Mapping[str, Any],
    *,
    index: int,
    dimensions: list[str],
    relationships_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    view_id = _required_string(row.get("id"), field=f"comparison_views[{index}].id")
    kind = _required_string(row.get("kind"), field=f"comparison_views[{index}].kind")
    if kind not in COMPARISON_VIEW_KINDS:
        raise OpalError(
            f"Campaign collection comparison_views[{index}].kind must be one of {sorted(COMPARISON_VIEW_KINDS)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    relationship_id = _required_string(
        row.get("relationship_id"),
        field=f"comparison_views[{index}].relationship_id",
    )
    relationship = relationships_by_id.get(relationship_id)
    if relationship is None:
        raise OpalError(
            f"Campaign collection comparison view {view_id!r} references unknown relationship_id {relationship_id!r}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    source_plot_name = _required_string(
        row.get("source_plot_name"),
        field=f"comparison_views[{index}].source_plot_name",
    )
    source_plot_kind = _required_string(
        row.get("source_plot_kind"),
        field=f"comparison_views[{index}].source_plot_kind",
    )
    expected_source_kind = _expected_source_plot_kind(kind)
    if source_plot_kind != expected_source_kind:
        raise OpalError(
            f"Campaign collection comparison view {view_id!r} kind={kind!r} requires "
            f"source_plot_kind={expected_source_kind!r}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    comparison_scope = _required_string(
        row.get("comparison_scope"),
        field=f"comparison_views[{index}].comparison_scope",
    )
    if comparison_scope not in COMPARISON_SCOPES:
        raise OpalError(
            f"Campaign collection comparison view {view_id!r} comparison_scope must be one of "
            f"{sorted(COMPARISON_SCOPES)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    group_key = _required_string(row.get("group_key"), field=f"comparison_views[{index}].group_key")
    _require_declared_dimensions([group_key], dimensions, field=f"comparison_views[{index}].group_key")
    match_filters = _string_mapping(row.get("match_filters"), field=f"comparison_views[{index}].match_filters")
    _require_declared_dimensions(list(match_filters), dimensions, field=f"comparison_views[{index}].match_filters")
    if kind == "paired_plot_gallery":
        summary = None
        metric = None
        cohort = None
    else:
        summary = _required_string(row.get("summary"), field=f"comparison_views[{index}].summary")
        metric = _required_string(row.get("metric"), field=f"comparison_views[{index}].metric")
        cohort = _required_string(row.get("cohort"), field=f"comparison_views[{index}].cohort")
        if kind == "vector_reference_mse_over_rounds_comparison" and metric != "reference_mse":
            raise OpalError(
                f"Campaign collection comparison view {view_id!r} requires metric='reference_mse'.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        if kind == "vector_heatmap_comparison" and metric != "selected_predicted_vector":
            raise OpalError(
                f"Campaign collection comparison view {view_id!r} requires metric='selected_predicted_vector'.",
                ExitCodes.CONTRACT_VIOLATION,
            )
    interval_kind = str(row.get("interval_kind") or "none").strip()
    if interval_kind not in INTERVAL_KINDS:
        raise OpalError(
            f"Campaign collection comparison view {view_id!r} interval_kind must be one of {sorted(INTERVAL_KINDS)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if kind == "paired_plot_gallery" and interval_kind != "none":
        raise OpalError(
            f"Campaign collection comparison view {view_id!r} paired_plot_gallery requires interval_kind='none'.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    confidence_level = row.get("confidence_level")
    if interval_kind == "student_t_mean_ci":
        if comparison_scope == "collection":
            raise OpalError(
                f"Campaign collection comparison view {view_id!r} student_t_mean_ci requires "
                "comparison_scope='comparison_set'.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        if summary != "mean":
            raise OpalError(
                f"Campaign collection comparison view {view_id!r} student_t_mean_ci requires summary='mean'.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        if not relationship.get("replicate_on"):
            raise OpalError(
                f"Campaign collection comparison view {view_id!r} student_t_mean_ci "
                "requires relationship replicate_on.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        confidence_level = _confidence_level(confidence_level, field=f"comparison_views[{index}].confidence_level")
    elif confidence_level is not None:
        raise OpalError(
            f"Campaign collection comparison view {view_id!r} declares confidence_level without a CI interval kind.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    payload = {
        "id": view_id,
        "label": str(row.get("label") or _pretty(view_id)),
        "kind": kind,
        "relationship_id": relationship_id,
        "source_plot_name": source_plot_name,
        "source_plot_kind": source_plot_kind,
        "comparison_scope": comparison_scope,
        "comparison_set_count": _comparison_set_count(relationship),
        "match_filters": match_filters,
        "group_key": group_key,
        "metric": metric,
        "cohort": cohort,
        "summary": summary,
        "interval_kind": interval_kind,
        "confidence_level": confidence_level,
        "relationship": dict(relationship),
    }
    interpretation_note = str(row.get("interpretation_note") or "").strip()
    if interpretation_note:
        payload["interpretation_note"] = interpretation_note
    return payload


def _expected_source_plot_kind(kind: str) -> str:
    if kind == "metric_over_rounds_comparison":
        return "metric_over_rounds"
    if kind in {"paired_plot_gallery", "vector_heatmap_comparison", "vector_reference_mse_over_rounds_comparison"}:
        return "vector_summary_heatmap"
    raise OpalError(
        f"Unsupported campaign collection comparison view kind: {kind!r}",
        ExitCodes.CONTRACT_VIOLATION,
    )


def _comparison_set_count(relationship: Mapping[str, Any]) -> int:
    replicate_on = {str(item) for item in relationship.get("replicate_on") or []}
    keys = set()
    for pair in relationship.get("pairs") or []:
        if not isinstance(pair, Mapping):
            continue
        match = pair.get("match") if isinstance(pair.get("match"), Mapping) else {}
        keys.add(tuple((str(key), str(value)) for key, value in sorted(match.items()) if str(key) not in replicate_on))
    return len(keys)


def _infer_role_dimension(
    *,
    dimensions: list[str],
    match_on: list[str],
    left_role: str,
    right_role: str,
    campaign_rows: list[dict[str, Any]],
    index: int,
) -> str:
    candidates = []
    for dimension in dimensions:
        if dimension in match_on:
            continue
        values = {str(row["metadata"].get(dimension)) for row in campaign_rows if dimension in row["metadata"]}
        if left_role in values and right_role in values:
            candidates.append(dimension)
    if len(candidates) != 1:
        raise OpalError(
            "Campaign collection relationship "
            f"{index} must declare role_dimension; inferred candidates were {candidates}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return candidates[0]


def _campaign_rows(campaigns: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    for campaign_model in campaigns:
        campaign = campaign_model.get("campaign") if isinstance(campaign_model, Mapping) else None
        if not isinstance(campaign, Mapping):
            continue
        slug = str(campaign.get("slug") or "").strip()
        if not slug:
            continue
        metadata = campaign.get("metadata") if isinstance(campaign.get("metadata"), Mapping) else {}
        raw_rows.append(
            {
                "slug": slug,
                "config_path": str(campaign.get("config_path") or "").strip(),
                "workdir": str(campaign.get("workdir") or "").strip(),
                "metadata": {str(key): value for key, value in metadata.items()},
            }
        )
    slug_counts = {row["slug"]: sum(1 for item in raw_rows if item["slug"] == row["slug"]) for row in raw_rows}
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        member_key = row["slug"]
        if slug_counts[row["slug"]] > 1:
            member_key = row["config_path"] or row["workdir"]
            if not member_key:
                raise OpalError(
                    "Campaign collection contains duplicate campaign slug "
                    f"{row['slug']!r} without config_path or workdir to disambiguate members.",
                    ExitCodes.CONTRACT_VIOLATION,
                )
        rows.append({"slug": row["slug"], "member_key": member_key, "metadata": row["metadata"]})
    return rows


def _dimension_rows(value: Any) -> list[dict[str, str]]:
    rows = _mapping_list(value, field="dimensions")
    if not rows:
        raise OpalError("Campaign collection field dimensions must not be empty.", ExitCodes.CONTRACT_VIOLATION)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    allowed = {"id", "label", "role", "description"}
    for index, row in enumerate(rows):
        unknown = sorted(str(key) for key in row if str(key) not in allowed)
        if unknown:
            raise OpalError(
                f"Campaign collection field dimensions[{index}] has unsupported key(s): {unknown}.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        dimension_id = _required_string(row.get("id"), field=f"dimensions[{index}].id")
        if dimension_id in seen:
            raise OpalError(
                f"Campaign collection field dimensions has duplicate id: {dimension_id!r}.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        seen.add(dimension_id)
        entry = {
            "id": dimension_id,
            "label": str(row.get("label") or _pretty(dimension_id)),
        }
        for key in ("role", "description"):
            if row.get(key) not in (None, ""):
                entry[key] = str(row[key])
        out.append(entry)
    return out


def _mapping_list(value: Any, *, field: str) -> list[Mapping[str, Any]]:
    if value in (None, ""):
        return []
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise OpalError(f"Campaign collection field {field} must be a list of mappings.", ExitCodes.CONTRACT_VIOLATION)
    return value


def _string_mapping(value: Any, *, field: str) -> dict[str, str]:
    if value in (None, ""):
        return {}
    if not isinstance(value, Mapping):
        raise OpalError(f"Campaign collection field {field} must be a mapping.", ExitCodes.CONTRACT_VIOLATION)
    out = {str(key).strip(): str(item).strip() for key, item in value.items()}
    if any(not key or not item for key, item in out.items()):
        raise OpalError(
            f"Campaign collection field {field} must map non-empty strings to non-empty strings.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return out


def _string_list(value: Any, *, field: str, allow_empty: bool) -> list[str]:
    if value in (None, "") and allow_empty:
        return []
    if not isinstance(value, list) or any(not str(item).strip() for item in value):
        raise OpalError(
            f"Campaign collection field {field} must be a list of non-empty strings.", ExitCodes.CONTRACT_VIOLATION
        )
    items = [str(item).strip() for item in value]
    if not allow_empty and not items:
        raise OpalError(f"Campaign collection field {field} must not be empty.", ExitCodes.CONTRACT_VIOLATION)
    return items


def _require_unique_ids(rows: Iterable[Mapping[str, Any]], *, field: str) -> None:
    ids = [str(row.get("id") or "") for row in rows]
    duplicates = sorted({item for item in ids if item and ids.count(item) > 1})
    if duplicates:
        raise OpalError(
            f"Campaign collection field {field} has duplicate id(s): {duplicates}.",
            ExitCodes.CONTRACT_VIOLATION,
        )


def _required_string(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise OpalError(f"Campaign collection field {field} must be a non-empty string.", ExitCodes.CONTRACT_VIOLATION)
    return text


def _require_declared_dimensions(values: list[str], dimensions: list[str], *, field: str) -> None:
    missing = [value for value in values if value not in set(dimensions)]
    if missing:
        raise OpalError(
            f"Campaign collection field {field} references undeclared dimension(s): {missing}.",
            ExitCodes.CONTRACT_VIOLATION,
        )


def _confidence_level(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OpalError(
            f"Campaign collection field {field} must be a numeric confidence level.",
            ExitCodes.CONTRACT_VIOLATION,
        ) from exc
    if not 0.0 < number < 1.0:
        raise OpalError(
            f"Campaign collection field {field} must be between 0 and 1.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return number


def _relationship_label(*, kind: str, role_dimension: str) -> str:
    return f"{_pretty(kind)} by {_pretty(role_dimension).lower()}"


def _pretty(value: Any) -> str:
    aliases = {
        "probe_label_family_id": "label family",
        "probe_oracle_kind": "label oracle kind",
        "probe_seed": "seed",
        "probe_split_id": "label split",
        "probe_target": "target",
    }
    text = str(value).strip()
    return aliases.get(text, text.replace("_", " ").capitalize())
