"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/api.py

Task-oriented public API for junction planning and publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dnadesign.junction.contracts.plan import JunctionPlan
from dnadesign.junction.contracts.request import JunctionRequest, canonical_request_bytes, load_request
from dnadesign.junction.design.planner import design_junction
from dnadesign.junction.errors import JunctionDesignError
from dnadesign.junction.publication import BundleVerification, PublishedJunctionBundle
from dnadesign.junction.publication.verify import _verify_published_bundle
from dnadesign.junction.publication.writer import _preflight_bundle_destination, _publish_bundle


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
