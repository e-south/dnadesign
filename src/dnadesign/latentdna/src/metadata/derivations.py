"""Config-driven metadata derivation helpers for latentdna."""

from __future__ import annotations

import re
from typing import Any

from ..contracts.errors import ContractViolationError
from ..contracts.workspace import MetadataDerivationConfig


def _normalize(value: object, *, mode: str | None) -> object:
    if mode is None or value is None:
        return value
    text = str(value)
    if mode == "lower":
        return text.lower()
    if mode == "upper":
        return text.upper()
    raise ContractViolationError(f"unsupported metadata normalization mode: {mode!r}")


def derive_metadata_value(row: dict[str, Any], derivation: MetadataDerivationConfig) -> object:
    if derivation.kind == "copy":
        return row.get(derivation.source)
    if derivation.kind == "regex_capture":
        source_value = row.get(derivation.source)
        if source_value is None:
            return derivation.default
        match = re.search(derivation.pattern, str(source_value))
        if match is None:
            return derivation.default
        return _normalize(match.group(derivation.group), mode=derivation.normalize)
    if derivation.kind == "map_values":
        source_value = row.get(derivation.source)
        if source_value is None:
            return derivation.default
        return derivation.mapping.get(str(source_value), derivation.default)
    if derivation.kind == "coalesce":
        for source_name in derivation.sources:
            value = row.get(source_name)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return derivation.default
    if derivation.kind == "constant":
        return derivation.value
    raise ContractViolationError(f"unsupported metadata derivation kind: {derivation.kind}")
