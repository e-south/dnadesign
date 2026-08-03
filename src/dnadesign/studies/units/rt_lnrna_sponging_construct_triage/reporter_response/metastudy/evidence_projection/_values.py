"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/evidence_projection/_values.py

Strict JSON-shaped value decoding for offline profile projections.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields


def strict_dataclass(value: object, cls: type[object]) -> dict[str, object]:
    declared_fields = fields(cls)
    payload = strict_object(value, label=cls.__name__, fields={item.name for item in declared_fields})
    return {item.name: payload[item.name] for item in declared_fields if item.init}


def strict_object(value: object, *, label: str, fields: set[str]) -> dict[str, object]:
    payload = mapping(value, label=label)
    if set(payload) != fields:
        raise ValueError(f"{label} fields do not match the exact contract")
    return payload


def mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def object_list(value: object, *, label: str) -> tuple[dict[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an array")
    return tuple(mapping(item, label=f"{label}[]") for item in value)


def number_tuple(value: object, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return tuple(value)


def required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


__all__ = [
    "mapping",
    "number_tuple",
    "object_list",
    "required_text",
    "strict_dataclass",
    "strict_object",
]
