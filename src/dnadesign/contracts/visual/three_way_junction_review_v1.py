"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/visual/three_way_junction_review_v1.py

Neutral review evidence for one three-way-junction assembly target.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common import PositiveLengthSpan, VisualContractModel

_DNA = re.compile(r"^[ACGT]+$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _require_dna(value: str, *, field: str, optional: bool = False) -> str:
    if optional and value == "":
        return value
    if not _DNA.fullmatch(value):
        raise ValueError(f"{field} must be non-empty uppercase ACGT")
    return value


def _require_sha256(value: str) -> str:
    if not _SHA256.fullmatch(value):
        raise ValueError("value must use sha256:<64 lowercase hexadecimal characters>")
    return value


class ReviewSource(VisualContractModel):
    plan_schema: Literal["dnadesign.trijunction.plan.v1"]
    plan_id: str
    request_sha256: str
    algorithm: str = Field(min_length=1)

    _validate_plan_id = field_validator("plan_id")(_require_sha256)
    _validate_request_sha256 = field_validator("request_sha256")(_require_sha256)


class ReviewTarget(VisualContractModel):
    target_id: str = Field(min_length=1)
    pool_id: str = Field(min_length=1)
    sequence_5to3: str
    sequence_sha256: str

    @field_validator("sequence_5to3")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return _require_dna(value, field="target.sequence_5to3")

    _validate_sequence_sha256 = field_validator("sequence_sha256")(_require_sha256)

    @model_validator(mode="after")
    def _validate_digest(self) -> "ReviewTarget":
        observed = f"sha256:{hashlib.sha256(self.sequence_5to3.encode()).hexdigest()}"
        if self.sequence_sha256 != observed:
            raise ValueError("target.sequence_sha256 does not match target.sequence_5to3")
        return self


class FragmentGeometry(VisualContractModel):
    fragment_id: str = Field(min_length=1)
    index: int = Field(ge=0)
    role: Literal["first", "internal", "last"]
    domain_span: PositiveLengthSpan


class JunctionGeometry(VisualContractModel):
    junction_id: str = Field(min_length=1)
    toehold_span: PositiveLengthSpan
    left_fragment_id: str = Field(min_length=1)
    right_fragment_id: str = Field(min_length=1)
    toehold: str
    toehold_complement: str
    barcode: str
    barcode_complement: str
    complement_nick_geometry_valid: Literal[True]
    complement_end_preparation: Literal["vendor_5_prime_phosphate", "downstream_phosphorylation"]

    @field_validator("toehold", "toehold_complement", "barcode", "barcode_complement")
    @classmethod
    def _validate_sequence(cls, value: str, info) -> str:
        return _require_dna(value, field=f"junction.{info.field_name}")

    @model_validator(mode="after")
    def _validate_complements(self) -> "JunctionGeometry":
        if self.toehold_complement != _reverse_complement(self.toehold):
            raise ValueError("toehold_complement must be the reverse complement of toehold")
        if self.barcode_complement != _reverse_complement(self.barcode):
            raise ValueError("barcode_complement must be the reverse complement of barcode")
        if self.toehold_span.end - self.toehold_span.start != len(self.toehold):
            raise ValueError("toehold_span length must equal the toehold length")
        return self


class AssemblyGeometry(VisualContractModel):
    fragments: tuple[FragmentGeometry, ...] = Field(min_length=2)
    junctions: tuple[JunctionGeometry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_graph(self) -> "AssemblyGeometry":
        expected_indices = list(range(len(self.fragments)))
        observed_indices = [fragment.index for fragment in self.fragments]
        if observed_indices != expected_indices:
            raise ValueError("geometry.fragments must be ordered with contiguous zero-based indices")
        expected_roles = ["first", *(["internal"] * (len(self.fragments) - 2)), "last"]
        if [fragment.role for fragment in self.fragments] != expected_roles:
            raise ValueError("geometry.fragments roles must be first, zero or more internal, then last")
        if len(self.junctions) != len(self.fragments) - 1:
            raise ValueError("geometry.junctions must contain exactly one junction between adjacent fragments")
        fragment_ids = [fragment.fragment_id for fragment in self.fragments]
        if len(fragment_ids) != len(set(fragment_ids)):
            raise ValueError("geometry.fragments fragment_id values must be unique")
        junction_ids = [junction.junction_id for junction in self.junctions]
        if len(junction_ids) != len(set(junction_ids)):
            raise ValueError("geometry.junctions junction_id values must be unique")
        for index, junction in enumerate(self.junctions):
            if junction.left_fragment_id != fragment_ids[index]:
                raise ValueError("junction.left_fragment_id must reference the adjacent left fragment")
            if junction.right_fragment_id != fragment_ids[index + 1]:
                raise ValueError("junction.right_fragment_id must reference the adjacent right fragment")
        return self


class FragmentStrands(VisualContractModel):
    fragment_id: str = Field(min_length=1)
    role: Literal["first", "internal", "last"]
    incoming_junction_id: str | None
    outgoing_junction_id: str | None
    barcode_bearing_sequence_5to3: str
    complement_sequence_5to3: str

    @field_validator("barcode_bearing_sequence_5to3", "complement_sequence_5to3")
    @classmethod
    def _validate_sequence(cls, value: str, info) -> str:
        return _require_dna(value, field=f"strands.{info.field_name}")


class PrimerReview(VisualContractModel):
    direction: Literal["forward", "reverse"]
    binding_sequence_5to3: str
    five_prime_extension_5to3: str
    order_sequence_5to3: str
    target_binding_span: PositiveLengthSpan

    @field_validator("binding_sequence_5to3", "order_sequence_5to3")
    @classmethod
    def _validate_required_sequence(cls, value: str, info) -> str:
        return _require_dna(value, field=f"primer.{info.field_name}")

    @field_validator("five_prime_extension_5to3")
    @classmethod
    def _validate_optional_sequence(cls, value: str) -> str:
        return _require_dna(value, field="primer.five_prime_extension_5to3", optional=True)

    @model_validator(mode="after")
    def _validate_order_sequence(self) -> "PrimerReview":
        expected = self.five_prime_extension_5to3 + self.binding_sequence_5to3
        if self.order_sequence_5to3 != expected:
            raise ValueError("order_sequence_5to3 must equal five_prime_extension_5to3 + binding_sequence_5to3")
        if self.target_binding_span.end - self.target_binding_span.start != len(self.binding_sequence_5to3):
            raise ValueError("target_binding_span length must equal binding_sequence_5to3 length")
        return self


class RecoveryReview(VisualContractModel):
    mode: Literal["target_specific", "universal"]
    forward: PrimerReview
    reverse: PrimerReview
    first_fragment_id: str = Field(min_length=1)
    last_fragment_id: str = Field(min_length=1)
    expected_product_sequence_5to3: str
    extended_top_sequence_5to3: str
    extended_bottom_sequence_5to3: str

    @field_validator(
        "expected_product_sequence_5to3",
        "extended_top_sequence_5to3",
        "extended_bottom_sequence_5to3",
    )
    @classmethod
    def _validate_expected_product(cls, value: str) -> str:
        return _require_dna(value, field="recovery.expected_product_sequence_5to3")

    @model_validator(mode="after")
    def _validate_directions(self) -> "RecoveryReview":
        if self.forward.direction != "forward" or self.reverse.direction != "reverse":
            raise ValueError("recovery primer directions must be forward and reverse")
        return self


class PoolSearchReview(VisualContractModel):
    pool_id: str = Field(min_length=1)
    toehold_seed: int
    barcode_generation_seed: int
    barcode_subset_seed: int
    matching_seed: int
    locus_count: int = Field(ge=1)
    toehold_paths_evaluated: int = Field(ge=1)
    toehold_min_distance: float = Field(ge=0)
    toehold_mean_distance: float = Field(ge=0)
    toehold_rank_score: float = Field(ge=0)
    barcode_candidates_generated: int = Field(ge=1)
    barcode_forbidden_toehold_k: int = Field(ge=1)
    barcode_forbidden_barcode_k: int = Field(ge=1)
    barcode_subsets_evaluated: int = Field(ge=1)
    barcode_min_distance: float = Field(ge=0)
    barcode_mean_distance: float = Field(ge=0)
    barcode_rank_score: float = Field(ge=0)
    matchings_evaluated: int = Field(ge=1)
    matching_max_pairwise_lcs: int = Field(ge=0)
    thermodynamic_screening: Literal["not_run"]


class CheckSubject(VisualContractModel):
    kind: Literal["pool", "target"]
    id: str = Field(min_length=1)


class ReviewCheck(VisualContractModel):
    subject: CheckSubject
    check: str = Field(min_length=1)
    status: Literal["passed", "not_run"]
    detail: str = Field(min_length=1)


class ThreeWayJunctionReviewV1(VisualContractModel):
    """Exact, study-neutral evidence for a semantic four-panel QA view."""

    contract_kind: Literal["three_way_junction_review_v1"]
    source: ReviewSource
    target: ReviewTarget
    geometry: AssemblyGeometry
    strands: tuple[FragmentStrands, ...] = Field(min_length=2)
    recovery: RecoveryReview
    search: PoolSearchReview
    checks: tuple[ReviewCheck, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_consistency(self) -> "ThreeWayJunctionReviewV1":
        target = self.target.sequence_5to3
        fragments = self.geometry.fragments
        junctions = self.geometry.junctions
        if fragments[0].domain_span.start != 0 or fragments[-1].domain_span.end != len(target):
            raise ValueError("geometry must begin at target coordinate 0 and end at the target length")
        for index, junction in enumerate(junctions):
            left = fragments[index].domain_span
            right = fragments[index + 1].domain_span
            if left.end != junction.toehold_span.start or junction.toehold_span.end != right.start:
                raise ValueError("fragment domains and toehold spans must partition the target without gaps or overlap")
            observed = target[junction.toehold_span.start : junction.toehold_span.end]
            if observed != junction.toehold:
                raise ValueError("junction.toehold must equal the declared target toehold span")

        strand_ids = [strand.fragment_id for strand in self.strands]
        fragment_ids = [fragment.fragment_id for fragment in fragments]
        if strand_ids != fragment_ids:
            raise ValueError("strands must contain one ordered entry for every geometry fragment")
        junction_ids = [junction.junction_id for junction in junctions]
        for index, strand in enumerate(self.strands):
            if strand.role != fragments[index].role:
                raise ValueError("strand roles must match geometry fragment roles")
            expected_incoming = None if index == 0 else junction_ids[index - 1]
            expected_outgoing = None if index == len(self.strands) - 1 else junction_ids[index]
            if strand.incoming_junction_id != expected_incoming or strand.outgoing_junction_id != expected_outgoing:
                raise ValueError("strand junction links must match the adjacent geometry junctions")
            domain_span = fragments[index].domain_span
            domain = target[domain_span.start : domain_span.end]
            previous = None if index == 0 else junctions[index - 1]
            following = None if index == len(fragments) - 1 else junctions[index]
            if previous is None:
                if following is None:
                    raise ValueError("a three-way-junction review requires at least one junction")
                expected_barcode_bearing = domain + following.toehold + following.barcode
                expected_complement = _reverse_complement(domain)
            elif following is None:
                expected_barcode_bearing = _reverse_complement(previous.barcode) + domain
                expected_complement = _reverse_complement(domain) + previous.toehold_complement
            else:
                expected_barcode_bearing = (
                    _reverse_complement(previous.barcode) + domain + following.toehold + following.barcode
                )
                expected_complement = _reverse_complement(domain) + previous.toehold_complement
            if strand.barcode_bearing_sequence_5to3 != expected_barcode_bearing:
                raise ValueError(
                    f"strands[{index}] barcode-bearing sequence does not match target and junction evidence"
                )
            if strand.complement_sequence_5to3 != expected_complement:
                raise ValueError(f"strands[{index}] complement sequence does not match target and junction evidence")

        recovery = self.recovery
        if recovery.first_fragment_id != fragment_ids[0] or recovery.last_fragment_id != fragment_ids[-1]:
            raise ValueError("recovery terminal fragment references must match geometry")
        if recovery.expected_product_sequence_5to3 != target:
            raise ValueError("recovery.expected_product_sequence_5to3 must equal the target sequence")
        for primer in (recovery.forward, recovery.reverse):
            if primer.target_binding_span.end > len(target):
                raise ValueError("primer target_binding_span exceeds the target length")
            target_segment = target[primer.target_binding_span.start : primer.target_binding_span.end]
            expected_binding = target_segment if primer.direction == "forward" else _reverse_complement(target_segment)
            if primer.binding_sequence_5to3 != expected_binding:
                raise ValueError(f"recovery.{primer.direction} binding sequence does not match its target span")
        if recovery.forward.target_binding_span.start != 0:
            raise ValueError("recovery.forward must bind the target prefix")
        if recovery.reverse.target_binding_span.end != len(target):
            raise ValueError("recovery.reverse must bind the target suffix")
        expected_extended_top = (
            recovery.forward.five_prime_extension_5to3
            + target
            + _reverse_complement(recovery.reverse.five_prime_extension_5to3)
        )
        expected_extended_bottom = (
            recovery.reverse.five_prime_extension_5to3
            + _reverse_complement(target)
            + _reverse_complement(recovery.forward.five_prime_extension_5to3)
        )
        if recovery.extended_top_sequence_5to3 != expected_extended_top:
            raise ValueError("recovery.extended_top_sequence_5to3 does not match the declared primer extensions")
        if recovery.extended_bottom_sequence_5to3 != expected_extended_bottom:
            raise ValueError("recovery.extended_bottom_sequence_5to3 does not match the declared primer extensions")
        if recovery.extended_bottom_sequence_5to3 != _reverse_complement(recovery.extended_top_sequence_5to3):
            raise ValueError("recovery extended top and bottom sequences must be reverse complements")
        if self.search.pool_id != self.target.pool_id:
            raise ValueError("search.pool_id must match target.pool_id")
        check_keys = [(check.subject.kind, check.subject.id, check.check) for check in self.checks]
        if len(check_keys) != len(set(check_keys)):
            raise ValueError("check subject and name tuples must be unique")
        for check in self.checks:
            if check.subject.kind == "target" and check.subject.id != self.target.target_id:
                raise ValueError("target check subject id must match target.target_id")
            if check.subject.kind == "pool" and check.subject.id != self.target.pool_id:
                raise ValueError("pool check subject id must match target.pool_id")
            if check.check == "thermodynamic_screening" and check.status != "not_run":
                raise ValueError("thermodynamic_screening check status must be not_run")
        return self


__all__ = ["ThreeWayJunctionReviewV1"]
