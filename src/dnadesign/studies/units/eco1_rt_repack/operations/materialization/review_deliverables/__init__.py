"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/__init__.py

Eco1 review-deliverable materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.models import (
    MaterializedReviewDeliverables,
)

if TYPE_CHECKING:
    from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.pipeline import (
        materialize_review_deliverables as materialize_review_deliverables,
    )

__all__ = ["MaterializedReviewDeliverables", "materialize_review_deliverables"]


def __getattr__(name: str) -> Any:
    if name == "materialize_review_deliverables":
        from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.pipeline import (
            materialize_review_deliverables,
        )

        return materialize_review_deliverables
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
