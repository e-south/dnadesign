"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/esm_atlas/models.py

Typed ESM Atlas adapter models.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AtlasIssue:
    """A generic Atlas semantic-artifact validation issue."""

    check_id: str
    message: str
    path: str = ""


@dataclass(frozen=True)
class AtlasNormalizedRows:
    """Normalized rows from one Atlas protein lookup response."""

    profile_row: dict[str, object]
    protein_activation_rows: list[dict[str, object]]
    residue_activation_rows: list[dict[str, object]]
    feature_catalog_rows: list[dict[str, object]]
