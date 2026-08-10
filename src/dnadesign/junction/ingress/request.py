"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/ingress/request.py

Compile normalized sequence records into the canonical Junction request.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence

from dnadesign.junction.contracts.request import (
    REQUEST_SCHEMA,
    JunctionRequest,
    OrderPolicy,
    PlanningProfile,
    Primer,
    RecoveryPrimerMode,
    RecoveryPrimerPair,
    Target,
    canonical_request_bytes,
)
from dnadesign.junction.contracts.request.validation import (
    parse_recovery_primer_mode,
    require_identifier,
    require_int,
    require_optional_dna,
)
from dnadesign.junction.errors import JunctionConfigError
from dnadesign.junction.sequence import reverse_complement

from .sources import SequenceRecord


def request_from_sequences(
    records: Sequence[SequenceRecord],
    *,
    planning: PlanningProfile,
    order_policy: OrderPolicy,
    seed: int,
    primer_binding_length: int,
    assembly_group_id: str = "assembly-01",
    recovery_mode: RecoveryPrimerMode = "target_specific",
    forward_five_prime_extension: str = "",
    reverse_five_prime_extension: str = "",
) -> JunctionRequest:
    """Apply explicit planning and recovery policy to normalized target sequences."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise JunctionConfigError("request_from_sequences requires at least one SequenceRecord")
    if any(not isinstance(record, SequenceRecord) for record in records):
        raise JunctionConfigError("request_from_sequences accepts only SequenceRecord values")
    binding_length = require_int(primer_binding_length, context="primer_binding_length", minimum=1)
    group_id = require_identifier(assembly_group_id, context="assembly_group_id")
    mode = parse_recovery_primer_mode(recovery_mode, context="recovery_mode")
    forward_extension = require_optional_dna(
        forward_five_prime_extension,
        context="forward_five_prime_extension",
    )
    reverse_extension = require_optional_dna(
        reverse_five_prime_extension,
        context="reverse_five_prime_extension",
    )
    targets: list[Target] = []
    primer_pairs: set[tuple[str, str, str, str]] = set()
    for record in records:
        if len(record.sequence) < binding_length:
            raise JunctionConfigError(
                f"sequence record {record.id!r} is shorter than primer_binding_length {binding_length}"
            )
        forward = Primer(
            binding_sequence=record.sequence[:binding_length],
            five_prime_extension=forward_extension,
        )
        reverse = Primer(
            binding_sequence=reverse_complement(record.sequence[-binding_length:]),
            five_prime_extension=reverse_extension,
        )
        primer_pairs.add(
            (
                forward.binding_sequence,
                forward.five_prime_extension,
                reverse.binding_sequence,
                reverse.five_prime_extension,
            )
        )
        targets.append(
            Target(
                id=record.id,
                assembly_group_id=group_id,
                sequence=record.sequence,
                recovery_primers=RecoveryPrimerPair(mode=mode, forward=forward, reverse=reverse),
            )
        )
    if mode == "universal" and len(primer_pairs) != 1:
        raise JunctionConfigError(
            "universal recovery requires every input sequence to share the declared terminal primer-binding sites"
        )
    request = JunctionRequest(
        schema=REQUEST_SCHEMA,
        seed=seed,
        planning=planning,
        targets=tuple(targets),
        order_policy=order_policy,
    )
    canonical_request_bytes(request)
    return request


__all__ = ["request_from_sequences"]
