"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/contracts/plan.py

Immutable output contracts for a junction design.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class SelectedJunction:
    """One selected toehold and its matched external barcode."""

    junction_id: str
    target_id: str
    assembly_group_id: str
    locus_index: int
    candidate_offset: int
    start: int
    toehold: str
    barcode_id: str
    barcode: str


@dataclass(frozen=True, slots=True)
class AssemblyGroupSearchEvidence:
    """Reproducibility receipt for one assembly-group search."""

    assembly_group_id: str
    toehold_seed: int
    barcode_generation_seed: int
    barcode_subset_seed: int
    matching_seed: int
    locus_count: int
    toehold_paths_evaluated: int
    toehold_min_distance: float
    toehold_mean_distance: float
    toehold_rank_score: float
    barcode_candidates_generated: int
    barcode_forbidden_toehold_k: int
    barcode_forbidden_barcode_k: int
    barcode_subsets_evaluated: int
    barcode_min_distance: float
    barcode_mean_distance: float
    barcode_rank_score: float
    matchings_evaluated: int
    matching_max_pairwise_lcs: int
    thermodynamic_screening: Literal["not_run"]


@dataclass(frozen=True, slots=True)
class AssemblyGroupPlan:
    """Jointly optimized assignments for one assembly group."""

    assembly_group_id: str
    junctions: tuple[SelectedJunction, ...]
    search: AssemblyGroupSearchEvidence


@dataclass(frozen=True, slots=True)
class FragmentPlan:
    """One paired fragment, with explicit terminal and strand roles."""

    fragment_id: str
    target_id: str
    assembly_group_id: str
    index: int
    role: str
    domain_start: int
    domain_end: int
    incoming_junction_id: str | None
    outgoing_junction_id: str | None
    barcode_bearing_strand: str
    complement_strand: str


@dataclass(frozen=True, slots=True)
class JunctionEvidence:
    """Sequence complementarity and strand layout at one selected junction."""

    junction_id: str
    left_fragment_id: str
    right_fragment_id: str
    toehold: str
    toehold_complement: str
    barcode: str
    barcode_complement: str
    complement_nick_sequence_layout_valid: bool
    complement_end_preparation: str


@dataclass(frozen=True, slots=True)
class RecoveryEvidence:
    """Exact primer-order and expected-sequence evidence for one target."""

    mode: str
    forward_binding_sequence: str
    forward_five_prime_extension: str
    forward_order_sequence: str
    forward_start: int
    forward_end: int
    reverse_binding_sequence: str
    reverse_five_prime_extension: str
    reverse_order_sequence: str
    reverse_start: int
    reverse_end: int
    first_fragment_id: str
    last_fragment_id: str
    expected_unextended_target: str
    extended_top_strand: str
    extended_bottom_strand: str


@dataclass(frozen=True, slots=True)
class TargetPlan:
    """Fragments and string checks for one exact submitted target."""

    target_id: str
    assembly_group_id: str
    assembly_kind: str
    target_sha256: str
    fragments: tuple[FragmentPlan, ...]
    junctions: tuple[JunctionEvidence, ...]
    recovery: RecoveryEvidence
    reconstructed_target: str
    reconstructed_complement: str


@dataclass(frozen=True, slots=True)
class OrderRecord:
    """One vendor-neutral orderable sequence row."""

    order_id: str
    target_ids: tuple[str, ...]
    assembly_group_id: str
    fragment_id: str | None
    role: str
    sequence: str
    sequence_sha256: str
    length: int
    five_prime_state: str
    synthesis_scale: str
    purification: str


@dataclass(frozen=True, slots=True)
class CheckSubject:
    """Exact plan entity to which one invariant result applies."""

    kind: Literal["assembly_group", "target"]
    id: str


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One machine-readable, structurally scoped invariant result."""

    check: str
    status: str
    subject: CheckSubject
    detail: str


@dataclass(frozen=True, slots=True)
class JunctionPlan:
    """Complete deterministic design and verification receipt."""

    schema: str
    algorithm: str
    request_sha256: str
    plan_id: str
    seed: int
    assembly_groups: tuple[AssemblyGroupPlan, ...]
    targets: tuple[TargetPlan, ...]
    orders: tuple[OrderRecord, ...]
    checks: tuple[CheckResult, ...]

    def to_mapping(self, *, include_plan_id: bool = True) -> dict[str, Any]:
        """Return the canonical JSON-compatible representation."""

        payload = _json_value(self)
        assert isinstance(payload, dict)
        if not include_plan_id:
            payload.pop("plan_id", None)
        return payload


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value
