"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/design/recovery.py

Assembly-group recovery validation and primer-order consolidation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING

from dnadesign.junction.contracts.plan import OrderRecord
from dnadesign.junction.errors import JunctionDesignError
from dnadesign.junction.sequence import reverse_complement

if TYPE_CHECKING:
    from dnadesign.junction.contracts.request import JunctionRequest, Target


def validate_recovery_set(request: JunctionRequest) -> None:
    """Reject recovery declarations that conflict within an assembly group."""

    targets_by_assembly_group: dict[str, list[Target]] = defaultdict(list)
    for target in request.targets:
        targets_by_assembly_group[target.assembly_group_id].append(target)

    for assembly_group_id, targets in targets_by_assembly_group.items():
        modes = {target.recovery_primers.mode for target in targets}
        if len(modes) != 1:
            raise JunctionDesignError(
                f"Assembly group '{assembly_group_id}' mixes recovery modes; choose one explicit mode per group."
            )
        mode = next(iter(modes))
        if mode == "universal":
            primer_pairs = {(target.recovery_primers.forward, target.recovery_primers.reverse) for target in targets}
            if len(primer_pairs) != 1:
                raise JunctionDesignError(
                    f"Assembly group '{assembly_group_id}' declares universal recovery but primer pairs differ."
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
                raise JunctionDesignError(
                    f"Target-specific recovery for '{target.id}' also resolves target(s): {joined}."
                )


def merge_universal_recovery_orders(
    request: JunctionRequest,
    orders: Iterable[OrderRecord],
) -> tuple[OrderRecord, ...]:
    """Collapse identical universal primer rows into one order per assembly group and direction."""

    universal_assembly_groups = {
        target.assembly_group_id for target in request.targets if target.recovery_primers.mode == "universal"
    }
    retained: list[OrderRecord] = []
    shared: dict[tuple[str, str, str, str, str, str], list[OrderRecord]] = defaultdict(list)
    for order in orders:
        if order.assembly_group_id in universal_assembly_groups and order.role in {
            "recovery_forward_primer",
            "recovery_reverse_primer",
        }:
            key = (
                order.assembly_group_id,
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
        assembly_group_id, role, *_ = key
        group = shared[key]
        template = min(group, key=lambda order: order.order_id)
        target_ids = tuple(sorted({target_id for order in group for target_id in order.target_ids}))
        direction = "forward" if role == "recovery_forward_primer" else "reverse"
        retained.append(
            replace(
                template,
                order_id=f"{assembly_group_id}:universal-recovery-{direction}",
                target_ids=target_ids,
            )
        )
    return tuple(sorted(retained, key=lambda order: (order.assembly_group_id, order.target_ids, order.order_id)))
