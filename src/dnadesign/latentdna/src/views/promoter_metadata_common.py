"""Shared helpers for LatentDNA promoter metadata derivations."""

from __future__ import annotations

import json

from ..contracts.errors import ContractViolationError


def normalize_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def canonical_regulator_name(value: object) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    token = text.split("_", 1)[0].strip()
    if not token:
        return None
    return {
        "baer": "baeR",
        "background": "background",
        "background_only": "background",
        "cpxr": "cpxR",
        "control": "control",
        "lexa": "lexA",
    }.get(token.lower(), token)


def normalized_regulators(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        values = value
    else:
        values = [value]
    normalized = sorted(
        {canonical_regulator_name(item) for item in values if canonical_regulator_name(item) is not None},
        key=str.casefold,
    )
    return [str(item) for item in normalized]


def coerce_list_of_dict_entries(value: object, *, field_name: str) -> list[dict[str, object]]:
    if value is None:
        return []
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - malformed payloads are caught by callers
            raise ContractViolationError(f"{field_name} must be valid JSON when encoded as text") from exc
    if not isinstance(value, list) and hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            value = converted
    if not isinstance(value, list):
        raise ContractViolationError(f"{field_name} must decode to a list of dict entries")
    entries: list[dict[str, object]] = []
    for item in value:
        if hasattr(item, "as_py"):
            item = item.as_py()
        if not isinstance(item, dict):
            raise ContractViolationError(f"{field_name} entries must be dictionaries")
        entries.append(dict(item))
    return entries
