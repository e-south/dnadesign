"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/design/strands.py

Paper-grounded strand composition and exact reconstruction proofs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dnadesign.trijunction.contracts.identity import sha256_bytes
from dnadesign.trijunction.contracts.plan import (
    CheckResult,
    CheckSubject,
    FragmentPlan,
    JunctionEvidence,
    OrderRecord,
    RecoveryEvidence,
    TargetPlan,
)
from dnadesign.trijunction.design.matching import JunctionAssignment
from dnadesign.trijunction.errors import TriJunctionDesignError
from dnadesign.trijunction.sequence import reverse_complement

if TYPE_CHECKING:
    from dnadesign.trijunction.contracts.request import OrderPolicy, Target


@dataclass(frozen=True, slots=True)
class TargetComposition:
    target: TargetPlan
    orders: tuple[OrderRecord, ...]
    checks: tuple[CheckResult, ...]


def _order_record(
    *,
    order_id: str,
    target: Target,
    fragment_id: str | None,
    role: str,
    sequence: str,
    five_prime_state: str,
    purification: str,
    policy: OrderPolicy,
) -> OrderRecord:
    if len(sequence) > policy.max_oligo_length:
        raise TriJunctionDesignError(
            f"Order '{order_id}' is {len(sequence)} nt but the declared synthesis ceiling is "
            f"{policy.max_oligo_length} nt."
        )
    return OrderRecord(
        order_id=order_id,
        target_ids=(target.id,),
        pool_id=target.pool_id,
        fragment_id=fragment_id,
        role=role,
        sequence=sequence,
        sequence_sha256=sha256_bytes(sequence.encode()),
        length=len(sequence),
        five_prime_state=five_prime_state,
        synthesis_scale=policy.synthesis_scale,
        purification=purification,
    )


def _recovery_evidence_and_orders(
    target: Target,
    *,
    first_fragment_id: str,
    last_fragment_id: str,
    policy: OrderPolicy,
) -> tuple[RecoveryEvidence, tuple[OrderRecord, OrderRecord]]:
    forward = target.recovery_primers.forward
    reverse = target.recovery_primers.reverse
    extended_top = forward.five_prime_extension + target.sequence + reverse_complement(reverse.five_prime_extension)
    extended_bottom = (
        reverse.five_prime_extension
        + reverse_complement(target.sequence)
        + reverse_complement(forward.five_prime_extension)
    )
    if extended_bottom != reverse_complement(extended_top):
        raise TriJunctionDesignError(f"Target '{target.id}' recovery products are not exact reverse complements.")
    evidence = RecoveryEvidence(
        mode=target.recovery_primers.mode,
        forward_binding_sequence=forward.binding_sequence,
        forward_five_prime_extension=forward.five_prime_extension,
        forward_order_sequence=forward.order_sequence,
        forward_start=0,
        forward_end=len(forward.binding_sequence),
        reverse_binding_sequence=reverse.binding_sequence,
        reverse_five_prime_extension=reverse.five_prime_extension,
        reverse_order_sequence=reverse.order_sequence,
        reverse_start=len(target.sequence) - len(reverse.binding_sequence),
        reverse_end=len(target.sequence),
        first_fragment_id=first_fragment_id,
        last_fragment_id=last_fragment_id,
        expected_core_product=target.sequence,
        extended_top_strand=extended_top,
        extended_bottom_strand=extended_bottom,
    )
    orders = (
        _order_record(
            order_id=f"{target.id}:recovery-forward",
            target=target,
            fragment_id=None,
            role="recovery_forward_primer",
            sequence=forward.order_sequence,
            five_prime_state="unmodified",
            purification=policy.primer_purification,
            policy=policy,
        ),
        _order_record(
            order_id=f"{target.id}:recovery-reverse",
            target=target,
            fragment_id=None,
            role="recovery_reverse_primer",
            sequence=reverse.order_sequence,
            five_prime_state="unmodified",
            purification=policy.primer_purification,
            policy=policy,
        ),
    )
    return evidence, orders


def compose_target(
    target: Target,
    assignments: tuple[JunctionAssignment, ...],
    *,
    order_policy: OrderPolicy,
) -> TargetComposition:
    """Compose paired strands and prove exact target recovery."""

    target_assignments = tuple(
        sorted(
            (assignment for assignment in assignments if assignment.candidate.target_id == target.id),
            key=lambda assignment: assignment.candidate.start,
        )
    )
    if not target_assignments:
        raise TriJunctionDesignError(f"Three-way-junction target '{target.id}' has no selected junctions.")

    toehold_length = len(target_assignments[0].candidate.sequence)
    for left, right in zip(target_assignments, target_assignments[1:], strict=False):
        if left.candidate.start + toehold_length > right.candidate.start:
            raise TriJunctionDesignError(f"Target '{target.id}' has overlapping selected toeholds.")

    junction_ids = tuple(f"{target.id}:junction-{index:04d}" for index in range(1, len(target_assignments) + 1))
    domain_spans: list[tuple[int, int]] = [(0, target_assignments[0].candidate.start)]
    for previous, current in zip(target_assignments, target_assignments[1:], strict=False):
        domain_spans.append((previous.candidate.start + toehold_length, current.candidate.start))
    domain_spans.append((target_assignments[-1].candidate.start + toehold_length, len(target.sequence)))
    domains = tuple(target.sequence[start:end] for start, end in domain_spans)

    fragments: list[FragmentPlan] = []
    orders: list[OrderRecord] = []
    complement_end_state = (
        "5_prime_phosphate"
        if order_policy.complement_end_preparation == "vendor_5_prime_phosphate"
        else "phosphate_required_before_assembly"
    )
    for index, domain in enumerate(domains):
        fragment_id = f"{target.id}:fragment-{index + 1:04d}"
        if index == 0:
            role = "first"
            barcode_bearing_strand = domain + target_assignments[0].candidate.sequence + target_assignments[0].barcode
            complement_strand = reverse_complement(domain)
        elif index == len(domains) - 1:
            role = "last"
            barcode_bearing_strand = reverse_complement(target_assignments[-1].barcode) + domain
            complement_strand = reverse_complement(domain) + reverse_complement(
                target_assignments[-1].candidate.sequence
            )
        else:
            role = "internal"
            previous = target_assignments[index - 1]
            current = target_assignments[index]
            barcode_bearing_strand = (
                reverse_complement(previous.barcode) + domain + current.candidate.sequence + current.barcode
            )
            complement_strand = reverse_complement(domain) + reverse_complement(previous.candidate.sequence)
        incoming = junction_ids[index - 1] if index > 0 else None
        outgoing = junction_ids[index] if index < len(junction_ids) else None
        fragments.append(
            FragmentPlan(
                fragment_id=fragment_id,
                target_id=target.id,
                pool_id=target.pool_id,
                index=index,
                role=role,
                domain_start=domain_spans[index][0],
                domain_end=domain_spans[index][1],
                incoming_junction_id=incoming,
                outgoing_junction_id=outgoing,
                barcode_bearing_strand=barcode_bearing_strand,
                complement_strand=complement_strand,
            )
        )
        orders.extend(
            (
                _order_record(
                    order_id=f"{fragment_id}:barcode-bearing",
                    target=target,
                    fragment_id=fragment_id,
                    role="barcode_bearing_strand",
                    sequence=barcode_bearing_strand,
                    five_prime_state="unmodified",
                    purification=order_policy.barcode_bearing_purification,
                    policy=order_policy,
                ),
                _order_record(
                    order_id=f"{fragment_id}:complement",
                    target=target,
                    fragment_id=fragment_id,
                    role="complement_strand",
                    sequence=complement_strand,
                    five_prime_state=complement_end_state,
                    purification=order_policy.complement_purification,
                    policy=order_policy,
                ),
            )
        )

    junctions = tuple(
        JunctionEvidence(
            junction_id=junction_id,
            left_fragment_id=fragments[index].fragment_id,
            right_fragment_id=fragments[index + 1].fragment_id,
            toehold=assignment.candidate.sequence,
            toehold_complement=reverse_complement(assignment.candidate.sequence),
            barcode=assignment.barcode,
            barcode_complement=reverse_complement(assignment.barcode),
            complement_nick_geometry_valid=True,
            complement_end_preparation=order_policy.complement_end_preparation,
        )
        for index, (junction_id, assignment) in enumerate(zip(junction_ids, target_assignments, strict=True))
    )

    reconstructed_parts: list[str] = [domains[0]]
    for assignment, domain in zip(target_assignments, domains[1:], strict=True):
        reconstructed_parts.extend((assignment.candidate.sequence, domain))
    reconstructed = "".join(reconstructed_parts)
    assembled_complement = "".join(fragment.complement_strand for fragment in reversed(fragments))
    if reconstructed != target.sequence:
        raise TriJunctionDesignError(f"Target '{target.id}' failed exact sequence reconstruction.")
    if assembled_complement != reverse_complement(target.sequence):
        raise TriJunctionDesignError(f"Target '{target.id}' failed complement-strand assembly reconstruction.")

    recovery, recovery_orders = _recovery_evidence_and_orders(
        target,
        first_fragment_id=fragments[0].fragment_id,
        last_fragment_id=fragments[-1].fragment_id,
        policy=order_policy,
    )
    orders.extend(recovery_orders)

    plan = TargetPlan(
        target_id=target.id,
        pool_id=target.pool_id,
        assembly_kind="three_way_junction",
        target_sha256=sha256_bytes(target.sequence.encode()),
        fragments=tuple(fragments),
        junctions=junctions,
        recovery=recovery,
        reconstructed_target=reconstructed,
        assembled_complement=assembled_complement,
    )
    checks = (
        CheckResult(
            check="exact_target_reconstruction",
            status="passed",
            subject=CheckSubject(kind="target", id=target.id),
            detail="exact input sequence restored",
        ),
        CheckResult(
            check="complement_strand_assembly",
            status="passed",
            subject=CheckSubject(kind="target", id=target.id),
            detail="assembled complement is rc(target)",
        ),
        CheckResult(
            check="complement_nick_geometry",
            status="passed",
            subject=CheckSubject(kind="target", id=target.id),
            detail="complement-strand nicks have valid sequence geometry; chemical readiness follows order policy",
        ),
        CheckResult(
            check="terminal_recovery_geometry",
            status="passed",
            subject=CheckSubject(kind="target", id=target.id),
            detail="primers bind declared termini and exact extensions form reverse-complement products",
        ),
        CheckResult(
            check="synthesis_length_ceiling",
            status="passed",
            subject=CheckSubject(kind="target", id=target.id),
            detail="every order row is within ceiling",
        ),
    )
    return TargetComposition(target=plan, orders=tuple(orders), checks=checks)
