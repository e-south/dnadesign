"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/publication/orders.py

Lossless vendor-neutral tabular projection of orderable oligos.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import io

from dnadesign.junction.contracts.plan import OrderRecord

ORDER_COLUMNS = (
    "order_id",
    "target_ids",
    "assembly_group_id",
    "fragment_id",
    "role",
    "sequence",
    "sequence_sha256",
    "length",
    "five_prime_state",
    "synthesis_scale",
    "purification",
)


def render_orders_tsv(orders: tuple[OrderRecord, ...]) -> bytes:
    """Render stable UTF-8 TSV without vendor-specific field names."""

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=ORDER_COLUMNS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for order in sorted(orders, key=lambda item: (item.assembly_group_id, item.target_ids, item.order_id)):
        writer.writerow(
            {
                "order_id": order.order_id,
                "target_ids": ",".join(order.target_ids),
                "assembly_group_id": order.assembly_group_id,
                "fragment_id": order.fragment_id or "",
                "role": order.role,
                "sequence": order.sequence,
                "sequence_sha256": order.sequence_sha256,
                "length": order.length,
                "five_prime_state": order.five_prime_state,
                "synthesis_scale": order.synthesis_scale,
                "purification": order.purification,
            }
        )
    return stream.getvalue().encode("utf-8")
