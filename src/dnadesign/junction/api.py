"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/api.py

Task-oriented public API for junction planning and publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from dnadesign.junction.contracts.plan import JunctionPlan
from dnadesign.junction.contracts.request import JunctionRequest, canonical_request_bytes, load_request
from dnadesign.junction.design.planner import design_junction
from dnadesign.junction.errors import JunctionConfigError, JunctionDesignError
from dnadesign.junction.ingress import SequenceRecord, load_sequence_records, request_from_sequences, sequence_record
from dnadesign.junction.publication import BundleVerification, PublishedJunctionBundle
from dnadesign.junction.publication.verify import _verify_published_bundle
from dnadesign.junction.publication.writer import _preflight_bundle_destination, _publish_bundle

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from dnadesign.junction.presentation.sequence_review import JunctionSequenceDissimilarityV1


@dataclass(frozen=True, slots=True)
class PlanSummary:
    status: str
    validation_scope: str
    request_sha256: str
    plan_id: str
    target_count: int
    assembly_group_count: int
    junction_count: int
    order_count: int
    thermodynamic_screening: Literal["not_run"]

    def to_mapping(self) -> dict[str, object]:
        return {
            "status": self.status,
            "validation_scope": self.validation_scope,
            "request_sha256": self.request_sha256,
            "plan_id": self.plan_id,
            "target_count": self.target_count,
            "assembly_group_count": self.assembly_group_count,
            "junction_count": self.junction_count,
            "order_count": self.order_count,
            "thermodynamic_screening": self.thermodynamic_screening,
        }


def _request(value: JunctionRequest | str | Path) -> JunctionRequest:
    if isinstance(value, JunctionRequest):
        canonical_request_bytes(value)
        return value
    return load_request(value)


def _design_v1(request: JunctionRequest) -> JunctionPlan:
    result = design_junction(request)
    screening_statuses = {assembly_group.search.thermodynamic_screening for assembly_group in result.assembly_groups}
    if screening_statuses != {"not_run"}:
        raise JunctionDesignError("junction v1 requires thermodynamic screening to remain explicitly not_run.")
    return result


def plan(value: JunctionRequest | str | Path) -> JunctionPlan:
    """Return the deterministic plan without writing files."""

    return _design_v1(_request(value))


def preflight(value: JunctionRequest | str | Path) -> PlanSummary:
    """Run the full design and return a summary without writing files."""

    result = plan(value)
    return PlanSummary(
        status="planned",
        validation_scope="string_only",
        request_sha256=result.request_sha256,
        plan_id=result.plan_id,
        target_count=len(result.targets),
        assembly_group_count=len(result.assembly_groups),
        junction_count=sum(len(target.junctions) for target in result.targets),
        order_count=len(result.orders),
        thermodynamic_screening="not_run",
    )


def build(
    value: JunctionRequest | str | Path,
    *,
    destination: str | Path,
) -> PublishedJunctionBundle:
    """Design and publish a verified bundle in a new destination."""

    request = _request(value)
    _preflight_bundle_destination(destination)
    result = _design_v1(request)
    return _publish_bundle(request, result, destination)


def verify(bundle: str | Path) -> BundleVerification:
    """Verify a bundle without relying on its original checkout."""

    return _verify_published_bundle(bundle)


def _sequence_dissimilarity_review(
    value: JunctionPlan | JunctionRequest | str | Path,
    *,
    assembly_group_id: str,
) -> JunctionSequenceDissimilarityV1:
    from dnadesign.junction.presentation.sequence_review import sequence_dissimilarity_contracts

    result = value if isinstance(value, JunctionPlan) else plan(value)
    reviews = sequence_dissimilarity_contracts(result)
    review = next((item for item in reviews if item.assembly_group_id == assembly_group_id), None)
    if review is None:
        available = ", ".join(item.assembly_group_id for item in reviews)
        raise JunctionConfigError(
            f"unknown assembly_group_id {assembly_group_id!r}; available assembly groups: {available}"
        )
    return review


def plot_sequence_dissimilarity(
    value: JunctionPlan | JunctionRequest | str | Path,
    *,
    assembly_group_id: str,
    junction_ids: Sequence[str] | None = None,
) -> Figure:
    """Plot the string comparisons used by one assembly group's search."""

    from dnadesign.junction.presentation.sequence_review.plot import (
        plot_sequence_dissimilarity as _plot_sequence_dissimilarity,
    )

    review = _sequence_dissimilarity_review(value, assembly_group_id=assembly_group_id)
    return _plot_sequence_dissimilarity(review, junction_ids=junction_ids)


def render_sequence_dissimilarity_svg(
    value: JunctionPlan | JunctionRequest | str | Path,
    *,
    assembly_group_id: str,
    junction_ids: Sequence[str] | None = None,
) -> bytes:
    """Render deterministic SVG bytes for one assembly group's string metrics."""

    from dnadesign.junction.presentation.sequence_review.plot import (
        render_sequence_dissimilarity_svg as _render_sequence_dissimilarity_svg,
    )

    review = _sequence_dissimilarity_review(value, assembly_group_id=assembly_group_id)
    return _render_sequence_dissimilarity_svg(review, junction_ids=junction_ids)


__all__ = [
    "PlanSummary",
    "SequenceRecord",
    "build",
    "load_sequence_records",
    "plan",
    "plot_sequence_dissimilarity",
    "preflight",
    "render_sequence_dissimilarity_svg",
    "request_from_sequences",
    "sequence_record",
    "verify",
]
