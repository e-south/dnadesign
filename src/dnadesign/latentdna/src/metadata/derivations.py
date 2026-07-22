"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/metadata/derivations.py

Config-driven metadata derivation helpers for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache
from importlib import import_module
from typing import Any

from ..contracts.errors import ContractViolationError
from ..contracts.workspace import MetadataDerivationConfig

AnnotationHandler = Callable[..., object]


@lru_cache(maxsize=32)
def _annotation_handler(handler_path: str) -> AnnotationHandler:
    module_name, function_name = handler_path.split(":", 1)
    try:
        module = import_module(module_name)
        handler = getattr(module, function_name)
    except (ImportError, AttributeError) as exc:
        raise ContractViolationError(f"metadata annotation handler cannot be loaded: {handler_path}") from exc
    if not callable(handler):
        raise ContractViolationError(f"metadata annotation handler is not callable: {handler_path}")
    return handler


def _normalize(value: object, *, mode: str | None) -> object:
    if mode is None or value is None:
        return value
    text = str(value)
    if mode == "lower":
        return text.lower()
    if mode == "upper":
        return text.upper()
    raise ContractViolationError(f"unsupported metadata normalization mode: {mode!r}")


def _nonempty_tokens(value: object, *, delimiter: str) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(delimiter) if part.strip()]


def _mean_numeric_token(value: object, *, delimiter: str, field_name: str) -> float | None:
    tokens = _nonempty_tokens(value, delimiter=delimiter)
    if not tokens:
        return None
    values: list[float] = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ContractViolationError(f"{field_name} contains a non-numeric metadata value: {token!r}") from exc
    return float(sum(values) / len(values))


def _single_categorical_token(value: object, *, delimiter: str, field_name: str) -> str | None:
    tokens = _nonempty_tokens(value, delimiter=delimiter)
    if not tokens:
        return None
    unique = sorted(set(tokens), key=str.casefold)
    if len(unique) != 1:
        raise ContractViolationError(f"{field_name} contains conflicting categorical values: {unique}")
    return unique[0]


def derive_metadata_value(row: dict[str, Any], derivation: MetadataDerivationConfig) -> object:
    if derivation.kind == "copy":
        return row.get(derivation.source)
    if derivation.kind == "token_presence":
        sources = [derivation.source] if derivation.source is not None else list(derivation.sources)
        for source in sources:
            if _nonempty_tokens(row.get(source), delimiter=derivation.delimiter):
                return derivation.present_value
        return derivation.absent_value
    if derivation.kind == "delimited_numeric_mean":
        return _mean_numeric_token(
            row.get(derivation.source),
            delimiter=derivation.delimiter,
            field_name=derivation.source,
        )
    if derivation.kind == "single_categorical_token":
        return _single_categorical_token(
            row.get(derivation.source),
            delimiter=derivation.delimiter,
            field_name=derivation.source,
        )
    if derivation.kind == "numeric_quantile_bin":
        value = _mean_numeric_token(
            row.get(derivation.source),
            delimiter=derivation.delimiter,
            field_name=derivation.source,
        )
        if value is None:
            return None
        for index, edge in enumerate(derivation.edges):
            if value <= float(edge):
                return derivation.labels[index]
        return derivation.labels[-1]
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
    if derivation.kind == "annotation":
        handler = _annotation_handler(derivation.handler)
        try:
            return handler(row, derive=derivation.derive)
        except ContractViolationError:
            if derivation.missing_policy == "null":
                return None
            raise
    raise ContractViolationError(f"unsupported metadata derivation kind: {derivation.kind}")
