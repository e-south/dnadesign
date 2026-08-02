"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/api.py

Task-oriented public API for TriJunction planning and publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.trijunction.contracts.plan import TriJunctionPlan
from dnadesign.trijunction.contracts.request import TriJunctionRequest, canonical_request_bytes, load_request
from dnadesign.trijunction.design.planner import design_trijunction
from dnadesign.trijunction.publication import BundleVerification, PublishedTriJunctionBundle
from dnadesign.trijunction.publication.verify import _verify_published_bundle
from dnadesign.trijunction.publication.writer import _preflight_bundle_destination, _publish_bundle


@dataclass(frozen=True, slots=True)
class PlanSummary:
    status: str
    validation_scope: str
    request_sha256: str
    plan_id: str
    target_count: int
    pool_count: int
    junction_count: int
    order_count: int
    thermodynamic_screening: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "status": self.status,
            "validation_scope": self.validation_scope,
            "request_sha256": self.request_sha256,
            "plan_id": self.plan_id,
            "target_count": self.target_count,
            "pool_count": self.pool_count,
            "junction_count": self.junction_count,
            "order_count": self.order_count,
            "thermodynamic_screening": self.thermodynamic_screening,
        }


def _request(value: TriJunctionRequest | str | Path) -> TriJunctionRequest:
    if isinstance(value, TriJunctionRequest):
        canonical_request_bytes(value)
        return value
    return load_request(value)


def plan(value: TriJunctionRequest | str | Path) -> TriJunctionPlan:
    """Return a pure, deterministic design without writing artifacts."""

    return design_trijunction(_request(value))


def preflight(value: TriJunctionRequest | str | Path) -> PlanSummary:
    """Run every design check and return a compact no-write receipt."""

    result = plan(value)
    screening_statuses = {pool.search.thermodynamic_screening for pool in result.pools}
    thermodynamic_screening = "not_run" if "not_run" in screening_statuses else "not_applicable"
    return PlanSummary(
        status="planned",
        validation_scope="string_only",
        request_sha256=result.request_sha256,
        plan_id=result.plan_id,
        target_count=len(result.targets),
        pool_count=len(result.pools),
        junction_count=sum(len(target.junctions) for target in result.targets),
        order_count=len(result.orders),
        thermodynamic_screening=thermodynamic_screening,
    )


def build(
    value: TriJunctionRequest | str | Path,
    *,
    destination: str | Path,
) -> PublishedTriJunctionBundle:
    """Design and publish one verified create-only bundle."""

    request = _request(value)
    _preflight_bundle_destination(destination)
    result = design_trijunction(request)
    return _publish_bundle(request, result, destination)


def verify(bundle: str | Path) -> BundleVerification:
    """Verify a bundle without relying on its original checkout."""

    return _verify_published_bundle(bundle)
