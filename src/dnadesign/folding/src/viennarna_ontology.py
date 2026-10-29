"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/viennarna_ontology.py

Shared dnadesign ontology helpers for ViennaRNA annotation publishing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

DEFAULT_COMPONENT_HUE = "#6F6F6F"


def component_token(owner_ids: tuple[str, ...]) -> str:
    if not owner_ids:
        return "unassigned"
    raw = owner_ids[0].split(".")[-1]
    return "".join(char if char.isalnum() or char == "_" else "_" for char in raw.lower())


def hue_for_owners(owner_ids: tuple[str, ...], *, palette: dict[str, str] | None = None) -> str:
    resolved_palette = palette or {}
    for owner_id in owner_ids:
        token = component_token((owner_id,))
        if token in resolved_palette:
            return resolved_palette[token]
    return DEFAULT_COMPONENT_HUE


def slug_token(value: str) -> str:
    token = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "section"


__all__ = ["DEFAULT_COMPONENT_HUE", "component_token", "hue_for_owners", "slug_token"]
