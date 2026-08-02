"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/contracts/request/codec.py

Strict mapping parser and stable serializer for TriJunction requests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from dnadesign.trijunction.contracts.identity import canonical_json_bytes

from ...errors import TriJunctionConfigError
from .limits import MAX_REQUEST_BYTES
from .model import (
    ComplementEndPreparation,
    OrderPolicy,
    PlanningProfile,
    Primer,
    RecoveryPrimerMode,
    RecoveryPrimerPair,
    Target,
    TriJunctionRequest,
)
from .validation import (
    parse_recovery_primer_mode,
    require_dna,
    require_exact_fields,
    require_fraction,
    require_identifier,
    require_int,
    require_mapping,
    require_optional_dna,
    require_plain_text,
)

_PLANNING_FIELDS = frozenset(
    {
        "oligo_length",
        "barcode_length",
        "toehold_length",
        "search_range",
        "toehold_search_iterations",
        "barcode_pool_factor",
        "barcode_generation_attempts",
        "barcode_toehold_k",
        "barcode_pair_k",
        "barcode_subset_iterations",
        "matching_iterations",
        "barcode_gc_min",
        "barcode_gc_max",
        "barcode_max_homopolymer",
    }
)
_RECOVERY_PRIMER_FIELDS = frozenset({"mode", "forward", "reverse"})
_PRIMER_FIELDS = frozenset({"binding_sequence", "five_prime_extension"})
_TARGET_FIELDS = frozenset({"id", "pool_id", "sequence", "recovery_primers"})
_ORDER_FIELDS = frozenset(
    {
        "synthesis_scale",
        "barcode_bearing_purification",
        "complement_purification",
        "primer_purification",
        "complement_end_preparation",
        "max_oligo_length",
    }
)
_REQUEST_FIELDS = frozenset({"schema", "seed", "planning", "targets", "order_policy"})


def _parse_planning(raw: object) -> PlanningProfile:
    planning_raw = require_mapping(raw, context="planning")
    require_exact_fields(planning_raw, required=_PLANNING_FIELDS, context="planning")
    return PlanningProfile(
        oligo_length=require_int(planning_raw["oligo_length"], context="planning.oligo_length", minimum=1),
        barcode_length=require_int(planning_raw["barcode_length"], context="planning.barcode_length", minimum=1),
        toehold_length=require_int(planning_raw["toehold_length"], context="planning.toehold_length", minimum=1),
        search_range=require_int(planning_raw["search_range"], context="planning.search_range", minimum=1),
        toehold_search_iterations=require_int(
            planning_raw["toehold_search_iterations"],
            context="planning.toehold_search_iterations",
            minimum=1,
        ),
        barcode_pool_factor=require_int(
            planning_raw["barcode_pool_factor"],
            context="planning.barcode_pool_factor",
            minimum=1,
        ),
        barcode_generation_attempts=require_int(
            planning_raw["barcode_generation_attempts"],
            context="planning.barcode_generation_attempts",
            minimum=1,
        ),
        barcode_toehold_k=require_int(
            planning_raw["barcode_toehold_k"],
            context="planning.barcode_toehold_k",
            minimum=1,
        ),
        barcode_pair_k=require_int(
            planning_raw["barcode_pair_k"],
            context="planning.barcode_pair_k",
            minimum=1,
        ),
        barcode_subset_iterations=require_int(
            planning_raw["barcode_subset_iterations"],
            context="planning.barcode_subset_iterations",
            minimum=1,
        ),
        matching_iterations=require_int(
            planning_raw["matching_iterations"],
            context="planning.matching_iterations",
            minimum=1,
        ),
        barcode_gc_min=require_fraction(planning_raw["barcode_gc_min"], context="planning.barcode_gc_min"),
        barcode_gc_max=require_fraction(planning_raw["barcode_gc_max"], context="planning.barcode_gc_max"),
        barcode_max_homopolymer=require_int(
            planning_raw["barcode_max_homopolymer"],
            context="planning.barcode_max_homopolymer",
            minimum=1,
        ),
    )


def _parse_primer(raw: object, *, context: str) -> Primer:
    primer_raw = require_mapping(raw, context=context)
    require_exact_fields(primer_raw, required=_PRIMER_FIELDS, context=context)
    return Primer(
        binding_sequence=require_dna(
            primer_raw["binding_sequence"],
            context=f"{context}.binding_sequence",
        ),
        five_prime_extension=require_optional_dna(
            primer_raw["five_prime_extension"],
            context=f"{context}.five_prime_extension",
        ),
    )


def _parse_targets(raw: object) -> tuple[Target, ...]:
    if not isinstance(raw, list):
        raise TriJunctionConfigError("targets must be a list")
    targets: list[Target] = []
    for index, target_value in enumerate(raw):
        context = f"targets[{index}]"
        target_raw = require_mapping(target_value, context=context)
        require_exact_fields(target_raw, required=_TARGET_FIELDS, context=context)
        primers_raw = require_mapping(target_raw["recovery_primers"], context=f"{context}.recovery_primers")
        require_exact_fields(
            primers_raw,
            required=_RECOVERY_PRIMER_FIELDS,
            context=f"{context}.recovery_primers",
        )
        targets.append(
            Target(
                id=require_identifier(target_raw["id"], context=f"{context}.id"),
                pool_id=require_identifier(target_raw["pool_id"], context=f"{context}.pool_id"),
                sequence=require_dna(target_raw["sequence"], context=f"{context}.sequence"),
                recovery_primers=RecoveryPrimerPair(
                    mode=cast(
                        RecoveryPrimerMode,
                        parse_recovery_primer_mode(
                            primers_raw["mode"],
                            context=f"{context}.recovery_primers.mode",
                        ),
                    ),
                    forward=_parse_primer(
                        primers_raw["forward"],
                        context=f"{context}.recovery_primers.forward",
                    ),
                    reverse=_parse_primer(
                        primers_raw["reverse"],
                        context=f"{context}.recovery_primers.reverse",
                    ),
                ),
            )
        )
    return tuple(targets)


def _parse_order_policy(raw: object) -> OrderPolicy:
    order_raw = require_mapping(raw, context="order_policy")
    require_exact_fields(order_raw, required=_ORDER_FIELDS, context="order_policy")
    complement_end_preparation = order_raw["complement_end_preparation"]
    if not isinstance(complement_end_preparation, str):
        raise TriJunctionConfigError("order_policy.complement_end_preparation must be a string")
    return OrderPolicy(
        synthesis_scale=require_plain_text(order_raw["synthesis_scale"], context="order_policy.synthesis_scale"),
        barcode_bearing_purification=require_plain_text(
            order_raw["barcode_bearing_purification"],
            context="order_policy.barcode_bearing_purification",
        ),
        complement_purification=require_plain_text(
            order_raw["complement_purification"],
            context="order_policy.complement_purification",
        ),
        primer_purification=require_plain_text(
            order_raw["primer_purification"], context="order_policy.primer_purification"
        ),
        complement_end_preparation=cast(ComplementEndPreparation, complement_end_preparation),
        max_oligo_length=require_int(order_raw["max_oligo_length"], context="order_policy.max_oligo_length", minimum=1),
    )


def parse_request(raw: Mapping[str, object]) -> TriJunctionRequest:
    """Validate an untrusted mapping and return its immutable canonical request."""

    request_raw = require_mapping(raw, context="request")
    require_exact_fields(request_raw, required=_REQUEST_FIELDS, context="request")

    planning = _parse_planning(request_raw["planning"])
    targets = _parse_targets(request_raw["targets"])
    order_policy = _parse_order_policy(request_raw["order_policy"])
    schema = request_raw["schema"]
    if not isinstance(schema, str):
        raise TriJunctionConfigError("schema must be a string")
    request = TriJunctionRequest(
        schema=schema,
        seed=require_int(request_raw["seed"], context="seed", minimum=0),
        planning=planning,
        targets=targets,
        order_policy=order_policy,
    )
    canonical_request_bytes(request)
    return request


def request_to_mapping(request: TriJunctionRequest) -> dict[str, object]:
    """Return the stable, JSON/YAML-safe mapping for a validated request."""

    return {
        "schema": request.schema,
        "seed": request.seed,
        "planning": {
            "oligo_length": request.planning.oligo_length,
            "barcode_length": request.planning.barcode_length,
            "toehold_length": request.planning.toehold_length,
            "search_range": request.planning.search_range,
            "toehold_search_iterations": request.planning.toehold_search_iterations,
            "barcode_pool_factor": request.planning.barcode_pool_factor,
            "barcode_generation_attempts": request.planning.barcode_generation_attempts,
            "barcode_toehold_k": request.planning.barcode_toehold_k,
            "barcode_pair_k": request.planning.barcode_pair_k,
            "barcode_subset_iterations": request.planning.barcode_subset_iterations,
            "matching_iterations": request.planning.matching_iterations,
            "barcode_gc_min": request.planning.barcode_gc_min,
            "barcode_gc_max": request.planning.barcode_gc_max,
            "barcode_max_homopolymer": request.planning.barcode_max_homopolymer,
        },
        "targets": [
            {
                "id": target.id,
                "pool_id": target.pool_id,
                "sequence": target.sequence,
                "recovery_primers": {
                    "mode": target.recovery_primers.mode,
                    "forward": {
                        "binding_sequence": target.recovery_primers.forward.binding_sequence,
                        "five_prime_extension": target.recovery_primers.forward.five_prime_extension,
                    },
                    "reverse": {
                        "binding_sequence": target.recovery_primers.reverse.binding_sequence,
                        "five_prime_extension": target.recovery_primers.reverse.five_prime_extension,
                    },
                },
            }
            for target in request.targets
        ],
        "order_policy": {
            "synthesis_scale": request.order_policy.synthesis_scale,
            "barcode_bearing_purification": request.order_policy.barcode_bearing_purification,
            "complement_purification": request.order_policy.complement_purification,
            "primer_purification": request.order_policy.primer_purification,
            "complement_end_preparation": request.order_policy.complement_end_preparation,
            "max_oligo_length": request.order_policy.max_oligo_length,
        },
    }


def canonical_request_bytes(request: TriJunctionRequest) -> bytes:
    """Serialize one request and enforce the shared input byte ceiling."""

    content = canonical_json_bytes(request_to_mapping(request))
    if len(content) > MAX_REQUEST_BYTES:
        raise TriJunctionConfigError(
            "TriJunction canonical request exceeds the "
            f"{MAX_REQUEST_BYTES}-byte input limit: observed {len(content)} bytes"
        )
    return content
