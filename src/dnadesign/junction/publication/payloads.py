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
from dnadesign.junction.presentation import render_review_contracts, render_sequence_dissimilarity_contracts
from dnadesign.junction.publication.fasta import (
    render_expected_products_fasta,
    render_orders_fasta,
    render_targets_fasta,
)
from dnadesign.junction.publication.orders import render_orders_tsv

ARTIFACT_PATHS = {
    "request": "request.json",
    "plan": "plan.json",
    "checks": "checks.json",
    "orders": "orders/oligos.tsv",
    "order_sequences": "sequences/oligos.fasta",
    "expected_products": "sequences/expected_pcr_products.fasta",
    "targets": "sequences/targets.fasta",
    "review": "views/three_way_junction_review.v1.json",
    "sequence_dissimilarity": "views/junction_sequence_dissimilarity.v1.json",
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
    if key == "order_sequences":
        return render_orders_fasta(plan)
    if key == "expected_products":
        return render_expected_products_fasta(plan)
    if key == "targets":
        return render_targets_fasta(request)
    if key == "review":
        return render_review_contracts(plan)
    if key == "sequence_dissimilarity":
        return render_sequence_dissimilarity_contracts(plan)
    raise KeyError(f"Unknown junction bundle artifact: {key!r}")
