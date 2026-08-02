"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/design/recovery.py

Physical-pool recovery validation and primer-order consolidation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING

from dnadesign.trijunction.contracts.plan import OrderRecord
from dnadesign.trijunction.errors import TriJunctionDesignError
from dnadesign.trijunction.sequence import reverse_complement

if TYPE_CHECKING:
    from dnadesign.trijunction.contracts.request import Target, TriJunctionRequest


def validate_recovery_set(request: TriJunctionRequest) -> None:
    """Reject recovery declarations that are ambiguous within a physical pool."""

    targets_by_pool: dict[str, list[Target]] = defaultdict(list)
    for target in request.targets:
        targets_by_pool[target.pool_id].append(target)

    for pool_id, targets in targets_by_pool.items():
        modes = {target.recovery_primers.mode for target in targets}
        if len(modes) != 1:
            raise TriJunctionDesignError(
                f"Physical pool '{pool_id}' mixes recovery modes; choose one explicit mode per pool."
            )
        mode = next(iter(modes))
        if mode == "universal":
            primer_pairs = {(target.recovery_primers.forward, target.recovery_primers.reverse) for target in targets}
            if len(primer_pairs) != 1:
                raise TriJunctionDesignError(
                    f"Physical pool '{pool_id}' declares universal recovery but primer pairs differ."
                )
            continue

        for target in targets:
            forward_binding = target.recovery_primers.forward.binding_sequence
            reverse_binding = reverse_complement(target.recovery_primers.reverse.binding_sequence)
            ambiguous = [
                other.id
                for other in targets
                if other.id != target.id
                and other.sequence.startswith(forward_binding)
                and other.sequence.endswith(reverse_binding)
            ]
            if ambiguous:
                joined = ", ".join(sorted(ambiguous))
                raise TriJunctionDesignError(
                    f"Target-specific recovery for '{target.id}' also resolves target(s): {joined}."
                )


def merge_universal_recovery_orders(
    request: TriJunctionRequest,
    orders: Iterable[OrderRecord],
) -> tuple[OrderRecord, ...]:
    """Collapse identical universal primer rows into one order per pool and direction."""

    universal_pools = {target.pool_id for target in request.targets if target.recovery_primers.mode == "universal"}
    retained: list[OrderRecord] = []
    shared: dict[tuple[str, str, str, str, str, str], list[OrderRecord]] = defaultdict(list)
    for order in orders:
        if order.pool_id in universal_pools and order.role in {
            "recovery_forward_primer",
            "recovery_reverse_primer",
        }:
            key = (
                order.pool_id,
                order.role,
                order.sequence,
                order.five_prime_state,
                order.synthesis_scale,
                order.purification,
            )
            shared[key].append(order)
        else:
            retained.append(order)

    for key in sorted(shared):
        pool_id, role, *_ = key
        group = shared[key]
        template = min(group, key=lambda order: order.order_id)
        target_ids = tuple(sorted({target_id for order in group for target_id in order.target_ids}))
        direction = "forward" if role == "recovery_forward_primer" else "reverse"
        retained.append(
            replace(
                template,
                order_id=f"{pool_id}:recovery-{direction}",
                target_ids=target_ids,
            )
        )
    return tuple(sorted(retained, key=lambda order: (order.pool_id, order.target_ids, order.order_id)))
