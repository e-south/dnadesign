"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/publication/fasta.py

Deterministic FASTA projections for Junction targets, orders, and products.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import io
from collections.abc import Iterable

from dnadesign.junction.contracts.plan import JunctionPlan
from dnadesign.junction.contracts.request import JunctionRequest

_LINE_WIDTH = 80


def _render_fasta(records: Iterable[tuple[str, str]]) -> bytes:
    stream = io.BytesIO()
    for header, sequence in records:
        stream.write(f">{header}\n".encode("ascii"))
        for start in range(0, len(sequence), _LINE_WIDTH):
            stream.write(sequence[start : start + _LINE_WIDTH].encode("ascii"))
            stream.write(b"\n")
    return stream.getvalue()


def render_targets_fasta(request: JunctionRequest) -> bytes:
    """Render normalized submitted targets as 5-prime-to-3-prime FASTA records."""

    return _render_fasta(
        (
            (
                f"{target.id} molecule=submitted_target assembly_group={target.assembly_group_id} "
                f"length={len(target.sequence)}",
                target.sequence,
            )
            for target in request.targets
        )
    )


def render_orders_fasta(plan: JunctionPlan) -> bytes:
    """Render every complete orderable sequence as one FASTA record."""

    return _render_fasta(
        (
            (
                f"{order.order_id} molecule=orderable_oligo role={order.role} "
                f"assembly_group={order.assembly_group_id} targets={','.join(order.target_ids)} "
                f"length={order.length}",
                order.sequence,
            )
            for order in sorted(plan.orders, key=lambda item: (item.assembly_group_id, item.target_ids, item.order_id))
        )
    )


def render_expected_products_fasta(plan: JunctionPlan) -> bytes:
    """Render each expected PCR top strand as one 5-prime-to-3-prime FASTA record."""

    return _render_fasta(
        (
            (
                f"{target.target_id} molecule=expected_pcr_product assembly_group={target.assembly_group_id} "
                f"length={len(target.recovery.extended_top_strand)}",
                target.recovery.extended_top_strand,
            )
            for target in sorted(plan.targets, key=lambda item: item.target_id)
        )
    )


__all__ = ["render_expected_products_fasta", "render_orders_fasta", "render_targets_fasta"]
