from __future__ import annotations

from typing import Any, Mapping

from ._support import mapping, sequence


def relationship_pair_membership(relationship: Mapping[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    rel = mapping(relationship)
    pairs = [pair for pair in sequence(rel.get("pairs")) if isinstance(pair, Mapping)]
    if not pairs:
        return {}
    role_dimension = str(rel.get("role_dimension") or rel.get("group_key") or "").strip()
    left_role = str(rel.get("left_role") or "left")
    right_role = str(rel.get("right_role") or "right")
    replicate_on = [str(item) for item in sequence(rel.get("replicate_on")) if str(item).strip()]
    membership: dict[str, list[dict[str, Any]]] = {}
    for index, pair in enumerate(pairs):
        match = mapping(pair.get("match"))
        pair_key = key_text(match) or f"pair={index + 1}"
        match_key = key_text({key: value for key, value in match.items() if key not in set(replicate_on)})
        replicate_key = key_text({key: value for key, value in match.items() if key in set(replicate_on)})
        common = {
            "relationship_kind": str(rel.get("relationship_kind") or rel.get("kind") or ""),
            "role_dimension": role_dimension,
            "match_key": match_key or pair_key,
            "replicate_key": replicate_key,
            "replicate_on": ",".join(replicate_on),
            "pair_key": pair_key,
            "comparison_unit_key": pair_key,
        }
        for side, role in (("left", left_role), ("right", right_role)):
            slug = str(pair.get(side) or "").strip()
            if slug:
                membership.setdefault(slug, []).append({**common, "comparison_role": role})
    return membership


def metadata_fields(metadata: Mapping[str, Any]) -> dict[str, str]:
    return {f"metadata__{key}": str(value) for key, value in metadata.items() if is_groupable_metadata_value(value)}


def key_text(values: Mapping[str, Any]) -> str:
    return "|".join(f"{key}={values[key]}" for key in sorted(values))


def is_groupable_metadata_value(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) and str(value).strip() != ""


def partition_signature(values: list[str]) -> tuple[int, ...]:
    seen: dict[str, int] = {}
    signature: list[int] = []
    for value in values:
        token = str(value)
        if token not in seen:
            seen[token] = len(seen)
        signature.append(seen[token])
    return tuple(signature)
