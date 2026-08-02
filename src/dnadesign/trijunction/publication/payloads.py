"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/publication/payloads.py

Canonical payload construction shared by publication and verification.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.trijunction.contracts.identity import canonical_json_bytes
from dnadesign.trijunction.contracts.plan import TriJunctionPlan
from dnadesign.trijunction.contracts.request import TriJunctionRequest, canonical_request_bytes
from dnadesign.trijunction.presentation import render_review_contracts
from dnadesign.trijunction.publication.orders import render_orders_tsv


def bundle_payloads(request: TriJunctionRequest, plan: TriJunctionPlan) -> dict[str, bytes]:
    checks: dict[str, Any] = {
        "schema": "dnadesign.trijunction.checks.v1",
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
    return {
        "request": canonical_request_bytes(request),
        "plan": canonical_json_bytes(plan.to_mapping()),
        "checks": canonical_json_bytes(checks),
        "orders": render_orders_tsv(plan.orders),
        "review": render_review_contracts(plan),
    }


ARTIFACT_PATHS = {
    "request": "request.json",
    "plan": "plan.json",
    "checks": "checks.json",
    "orders": "orders/oligos.tsv",
    "review": "views/three_way_junction_review.v1.json",
}
