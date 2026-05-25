"""Campaign-collection manifest contracts for OPAL notebooks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from ..core.utils import ExitCodes, OpalError

CAMPAIGN_COLLECTION_SCHEMA_VERSION = "opal.campaign_collection.v1"


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

    dimensions = _string_list(raw.get("dimensions"), field="dimensions", allow_empty=False)
    campaign_rows = _campaign_rows(campaigns)
    relationships = [
        _relationship_payload(row, index=index, dimensions=dimensions, campaign_rows=campaign_rows)
        for index, row in enumerate(_mapping_list(raw.get("relationships"), field="relationships"))
    ]
    return {
        "schema_version": CAMPAIGN_COLLECTION_SCHEMA_VERSION,
        "path": str(manifest_path),
        "dimensions": dimensions,
        "relationships": relationships,
        "relationship_count": len(relationships),
        "comparison_lenses": [_comparison_lens(row) for row in relationships],
    }


def _relationship_payload(
    row: Mapping[str, Any],
    *,
    index: int,
    dimensions: list[str],
    campaign_rows: list[dict[str, Any]],
) -> dict[str, Any]:
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
            f"Campaign collection relationship {kind!r} matched no campaign pairs.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return {
        "kind": kind,
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
                pairs.append(
                    {
                        "left": left_row["slug"],
                        "right": right_row["slug"],
                        "match": dict(zip(match_on, key, strict=True)),
                    }
                )
    return pairs


def _comparison_lens(relationship: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": relationship["kind"],
        "label": f"{_pretty(relationship['kind'])} by {_pretty(relationship['role_dimension']).lower()}",
        "group_key": relationship["role_dimension"],
        "role_dimension": relationship["role_dimension"],
        "left_role": relationship["left_role"],
        "right_role": relationship["right_role"],
        "match_on": list(relationship["match_on"]),
        "replicate_on": list(relationship.get("replicate_on") or []),
        "pair_count": int(relationship["pair_count"]),
        "pairs": list(relationship.get("pairs") or []),
    }


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
    rows: list[dict[str, Any]] = []
    for campaign_model in campaigns:
        campaign = campaign_model.get("campaign") if isinstance(campaign_model, Mapping) else None
        if not isinstance(campaign, Mapping):
            continue
        slug = str(campaign.get("slug") or "").strip()
        if not slug:
            continue
        metadata = campaign.get("metadata") if isinstance(campaign.get("metadata"), Mapping) else {}
        rows.append({"slug": slug, "metadata": {str(key): value for key, value in metadata.items()}})
    return rows


def _mapping_list(value: Any, *, field: str) -> list[Mapping[str, Any]]:
    if value in (None, ""):
        return []
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise OpalError(f"Campaign collection field {field} must be a list of mappings.", ExitCodes.CONTRACT_VIOLATION)
    return value


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
