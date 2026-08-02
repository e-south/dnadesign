"""Frozen values for the canonical TriJunction request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from ...errors import TriJunctionConfigError
from ...sequence import reverse_complement
from .validation import (
    COMPLEMENT_END_PREPARATIONS,
    RECOVERY_PRIMER_MODES,
    require_dna,
    require_fraction,
    require_identifier,
    require_int,
    require_optional_dna,
    require_plain_text,
)

REQUEST_SCHEMA = "dnadesign.trijunction.request.v1"

# Schema-v1 ceilings bound offline planning and verification work while leaving
# ample headroom above the literature-derived defaults.
MAX_TOEHOLD_SEARCH_ITERATIONS = 100_000
MAX_BARCODE_GENERATION_ATTEMPTS = 10_000_000
MAX_BARCODE_SUBSET_ITERATIONS = 100_000
MAX_MATCHING_ITERATIONS = 100_000

_PLANNING_BUDGET_CEILINGS = {
    "toehold_search_iterations": MAX_TOEHOLD_SEARCH_ITERATIONS,
    "barcode_generation_attempts": MAX_BARCODE_GENERATION_ATTEMPTS,
    "barcode_subset_iterations": MAX_BARCODE_SUBSET_ITERATIONS,
    "matching_iterations": MAX_MATCHING_ITERATIONS,
}

ComplementEndPreparation: TypeAlias = Literal[
    "vendor_5_prime_phosphate",
    "downstream_phosphorylation",
]
RecoveryPrimerMode: TypeAlias = Literal["target_specific", "universal"]


@dataclass(frozen=True, slots=True)
class PlanningProfile:
    """Search and sequence-layout budgets for a planning request."""

    oligo_length: int
    barcode_length: int
    toehold_length: int
    search_range: int
    toehold_search_iterations: int
    barcode_pool_factor: int
    barcode_generation_attempts: int
    barcode_toehold_k: int
    barcode_pair_k: int
    barcode_subset_iterations: int
    matching_iterations: int
    barcode_gc_min: float
    barcode_gc_max: float
    barcode_max_homopolymer: int

    def __post_init__(self) -> None:
        for field_name in (
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
            "barcode_max_homopolymer",
        ):
            require_int(getattr(self, field_name), context=f"planning.{field_name}", minimum=1)
        for field_name, maximum in _PLANNING_BUDGET_CEILINGS.items():
            if getattr(self, field_name) > maximum:
                raise TriJunctionConfigError(f"planning.{field_name} must not exceed {maximum} for {REQUEST_SCHEMA}")
        object.__setattr__(
            self,
            "barcode_gc_min",
            require_fraction(self.barcode_gc_min, context="planning.barcode_gc_min"),
        )
        object.__setattr__(
            self,
            "barcode_gc_max",
            require_fraction(self.barcode_gc_max, context="planning.barcode_gc_max"),
        )
        if self.barcode_gc_min > self.barcode_gc_max:
            raise TriJunctionConfigError("planning.barcode_gc_min must not exceed barcode_gc_max")
        if self.barcode_max_homopolymer > self.barcode_length:
            raise TriJunctionConfigError("planning.barcode_max_homopolymer must not exceed barcode_length")
        if self.barcode_pool_factor < 5:
            raise TriJunctionConfigError("planning.barcode_pool_factor must be at least 5")
        if self.toehold_length < 2:
            raise TriJunctionConfigError("planning.toehold_length must be at least 2")
        if self.barcode_toehold_k > min(self.barcode_length, self.toehold_length):
            raise TriJunctionConfigError("planning.barcode_toehold_k must not exceed barcode_length or toehold_length")
        if self.barcode_pair_k > self.barcode_length:
            raise TriJunctionConfigError("planning.barcode_pair_k must not exceed barcode_length")
        if self.barcode_pair_k <= self.barcode_toehold_k:
            raise TriJunctionConfigError("planning.barcode_pair_k must be greater than barcode_toehold_k")
        minimum_exclusive = 2 * self.barcode_length + self.toehold_length + self.search_range - 1
        if self.oligo_length <= minimum_exclusive:
            raise TriJunctionConfigError(
                "planning.oligo_length must be greater than 2 * barcode_length + toehold_length + search_range - 1"
            )


@dataclass(frozen=True, slots=True)
class Primer:
    """One ordered primer, separating target binding from an exact 5-prime extension."""

    binding_sequence: str
    five_prime_extension: str

    def __post_init__(self) -> None:
        require_dna(self.binding_sequence, context="primer.binding_sequence")
        require_optional_dna(self.five_prime_extension, context="primer.five_prime_extension")

    @property
    def order_sequence(self) -> str:
        """Return the complete 5-prime-to-3-prime sequence submitted for synthesis."""

        return self.five_prime_extension + self.binding_sequence


@dataclass(frozen=True, slots=True)
class RecoveryPrimerPair:
    """Forward and reverse primers used to recover one assembled target."""

    mode: RecoveryPrimerMode
    forward: Primer
    reverse: Primer

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in RECOVERY_PRIMER_MODES:
            allowed = ", ".join(sorted(RECOVERY_PRIMER_MODES))
            raise TriJunctionConfigError(f"recovery_primers.mode must be one of: {allowed}")
        if not isinstance(self.forward, Primer):
            raise TriJunctionConfigError("recovery_primers.forward must be a Primer value")
        if not isinstance(self.reverse, Primer):
            raise TriJunctionConfigError("recovery_primers.reverse must be a Primer value")


@dataclass(frozen=True, slots=True)
class Target:
    """One exact linear DNA target and its physical-pool identity."""

    id: str
    pool_id: str
    sequence: str
    recovery_primers: RecoveryPrimerPair

    def __post_init__(self) -> None:
        require_identifier(self.id, context="target.id")
        require_identifier(self.pool_id, context="target.pool_id")
        require_dna(self.sequence, context=f"target {self.id!r} sequence")
        if not isinstance(self.recovery_primers, RecoveryPrimerPair):
            raise TriJunctionConfigError("target.recovery_primers must be a RecoveryPrimerPair value")
        if not self.sequence.startswith(self.recovery_primers.forward.binding_sequence):
            raise TriJunctionConfigError(f"target {self.id!r} recovery forward primer must match the target prefix")
        reverse_binding = self.recovery_primers.reverse.binding_sequence
        reverse_suffix = reverse_complement(self.sequence[-len(reverse_binding) :])
        if reverse_binding != reverse_suffix:
            raise TriJunctionConfigError(
                f"target {self.id!r} recovery reverse primer must match the reverse-complemented target suffix"
            )


@dataclass(frozen=True, slots=True)
class OrderPolicy:
    """Vendor-facing ordering constraints."""

    synthesis_scale: str
    barcode_bearing_purification: str
    complement_purification: str
    primer_purification: str
    complement_end_preparation: ComplementEndPreparation
    max_oligo_length: int

    def __post_init__(self) -> None:
        for field_name in (
            "synthesis_scale",
            "barcode_bearing_purification",
            "complement_purification",
            "primer_purification",
        ):
            require_plain_text(getattr(self, field_name), context=f"order_policy.{field_name}")
        if (
            not isinstance(self.complement_end_preparation, str)
            or self.complement_end_preparation not in COMPLEMENT_END_PREPARATIONS
        ):
            allowed = ", ".join(sorted(COMPLEMENT_END_PREPARATIONS))
            raise TriJunctionConfigError(f"order_policy.complement_end_preparation must be one of: {allowed}")
        require_int(self.max_oligo_length, context="order_policy.max_oligo_length", minimum=1)


@dataclass(frozen=True, slots=True)
class TriJunctionRequest:
    """Canonical, validated TriJunction planning request."""

    schema: str
    seed: int
    planning: PlanningProfile
    targets: tuple[Target, ...]
    order_policy: OrderPolicy

    def __post_init__(self) -> None:
        if self.schema != REQUEST_SCHEMA:
            raise TriJunctionConfigError(f"schema must equal {REQUEST_SCHEMA!r}")
        require_int(self.seed, context="seed", minimum=0)
        if not isinstance(self.planning, PlanningProfile):
            raise TriJunctionConfigError("planning must be a PlanningProfile value")
        if not isinstance(self.order_policy, OrderPolicy):
            raise TriJunctionConfigError("order_policy must be an OrderPolicy value")
        if not isinstance(self.targets, tuple) or not self.targets:
            raise TriJunctionConfigError("request must contain at least one target")
        if any(not isinstance(target, Target) for target in self.targets):
            raise TriJunctionConfigError("targets must contain only Target values")
        maximum_expected_length = self.planning.oligo_length + self.planning.search_range - 1
        if self.order_policy.max_oligo_length < maximum_expected_length:
            raise TriJunctionConfigError(
                "order_policy.max_oligo_length must be at least planning.oligo_length + planning.search_range - 1"
            )
        for target in self.targets:
            for direction, primer in (
                ("forward", target.recovery_primers.forward),
                ("reverse", target.recovery_primers.reverse),
            ):
                if len(primer.order_sequence) > self.order_policy.max_oligo_length:
                    raise TriJunctionConfigError(
                        f"target {target.id!r} recovery {direction} primer is {len(primer.order_sequence)} nt but "
                        f"order_policy.max_oligo_length is {self.order_policy.max_oligo_length}"
                    )

        ids: set[str] = set()
        physical_sequences: set[tuple[str, str]] = set()
        for target in self.targets:
            if target.id in ids:
                raise TriJunctionConfigError(f"duplicate target id: {target.id!r}")
            ids.add(target.id)
            physical_key = (target.pool_id, target.sequence)
            if physical_key in physical_sequences:
                raise TriJunctionConfigError(f"duplicate sequence within physical pool {target.pool_id!r}")
            physical_sequences.add(physical_key)
        canonical_targets = tuple(sorted(self.targets, key=lambda target: target.id))
        object.__setattr__(self, "targets", canonical_targets)

    def to_mapping(self) -> dict[str, object]:
        """Return the stable, JSON/YAML-safe representation of this request."""

        from .codec import request_to_mapping

        return request_to_mapping(self)
