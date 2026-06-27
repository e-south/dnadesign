"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/models.py

Typed models for Eco1 review-deliverable materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaterializedReviewDeliverables:
    """Paths emitted by one Eco1 review-deliverables materialization pass."""

    manifest_path: Path
    notebook_path: Path
    deliverable_count: int
