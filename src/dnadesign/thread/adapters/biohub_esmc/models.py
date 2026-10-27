"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/biohub_esmc/models.py

Small value objects for normalized Biohub ESMC artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BiohubEsmcIssue:
    """One validation issue for Biohub ESMC artifacts."""

    check_id: str
    message: str
    path: str


@dataclass(frozen=True)
class BiohubEsmcNormalizedRows:
    """Normalized rows for one Biohub ESMC logits response."""

    profile_row: dict[str, object]
    protein_feature_rows: list[dict[str, object]]
    residue_feature_rows: list[dict[str, object]]
    feature_catalog_rows: list[dict[str, object]]
