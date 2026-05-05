"""Representation-family contract helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

RETIRED_REPRESENTATION_FAMILY_REPLACEMENTS = {
    "pooled_logits": "output_layer_mean",
}

RETIRED_REPRESENTATION_TERM_REPLACEMENTS = {
    "pooled_logits": "output_layer_mean",
    "pooled logits": "output-layer mean",
    "pooled-logits": "output-layer mean",
}


def retired_representation_term(value: str) -> tuple[str, str] | None:
    normalized = value.casefold()
    for retired, replacement in RETIRED_REPRESENTATION_TERM_REPLACEMENTS.items():
        if retired in normalized:
            return retired, replacement
    return None


def validate_representation_identity(value: str, *, owner: str) -> None:
    """Validate identifiers that become part of persisted representation identity."""

    retired = retired_representation_term(value)
    if retired is None:
        return
    retired_term, replacement = retired
    raise ValueError(f"{owner} uses retired representation term {retired_term!r}; use {replacement!r}")


def validate_representation_family_tags(tags: Mapping[str, Any], *, owner: str) -> None:
    """Validate representation-family tags that define LatentDNA geometry identity."""

    family = tags.get("family")
    if family is None:
        return
    if not isinstance(family, str):
        raise ValueError(f"{owner} tags.family must be a string")
    normalized_family = family.strip()
    replacement = RETIRED_REPRESENTATION_FAMILY_REPLACEMENTS.get(normalized_family)
    if replacement is not None:
        raise ValueError(
            f"{owner} uses retired representation family {normalized_family!r}; "
            f"use {replacement!r} for Infer output-layer mean vectors"
        )
