"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/publication/payloads.py

Canonical payload construction shared by publication and verification.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.junction.contracts.identity import canonical_json_bytes
from dnadesign.junction.contracts.plan import JunctionPlan
from dnadesign.junction.contracts.request import JunctionRequest, canonical_request_bytes
from dnadesign.junction.presentation import render_review_contracts
from dnadesign.junction.publication.orders import render_orders_tsv

ARTIFACT_PATHS = {
    "request": "request.json",
    "plan": "plan.json",
    "checks": "checks.json",
    "orders": "orders/oligos.tsv",
    "review": "views/three_way_junction_review.v1.json",
}


def render_artifact_bytes(key: str, request: JunctionRequest, plan: JunctionPlan) -> bytes:
    """Render one canonical bundle artifact without retaining its siblings."""

    if key == "request":
        return canonical_request_bytes(request)
    if key == "plan":
        return canonical_json_bytes(plan.to_mapping())
    if key == "checks":
        checks: dict[str, Any] = {
            "schema": "dnadesign.junction.checks.v1",
            "plan_id": plan.plan_id,
            "checks": [
                {
                    "check": check.check,
                    "status": check.status,
                    "subject": {"kind": check.subject.kind, "id": check.subject.id},
                    "detail": check.detail,
                }
                for check in plan.checks
            ],
        }
        return canonical_json_bytes(checks)
    if key == "orders":
        return render_orders_tsv(plan.orders)
    if key == "review":
        return render_review_contracts(plan)
    raise KeyError(f"Unknown junction bundle artifact: {key!r}")
