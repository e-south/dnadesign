"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/adapters/three_way_junction_review_v1.py

Adapter from neutral three-way-junction review evidence to Record v1.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from pydantic import ValidationError

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1
from dnadesign.contracts.visual.three_way_junction_review_v1 import JUNCTION_STRING_V1_ALGORITHM

from ..core import ContractError, Record, SchemaError
from ..core.pydantic_validation import format_validation_error
from ..core.record import Display


@dataclass(frozen=True)
class ThreeWayJunctionReviewV1Adapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str
    _source_receipts: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _search_receipts: dict[tuple[str, str], dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _assembly_group_check_evidence: dict[
        tuple[str, str],
        tuple[tuple[str, str, str, str, str], ...],
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _v1_complement_end_preparation: dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _target_identities: set[tuple[str, str]] = field(
        default_factory=set,
        init=False,
        repr=False,
        compare=False,
    )
    _grouped_sequence_identities: set[tuple[str, str, str]] = field(
        default_factory=set,
        init=False,
        repr=False,
        compare=False,
    )
    _fragment_identities: set[tuple[str, str]] = field(
        default_factory=set,
        init=False,
        repr=False,
        compare=False,
    )
    _junction_identities: set[tuple[str, str]] = field(
        default_factory=set,
        init=False,
        repr=False,
        compare=False,
    )
    _barcode_identities: set[tuple[str, str, str]] = field(
        default_factory=set,
        init=False,
        repr=False,
        compare=False,
    )
    _junction_counts: dict[tuple[str, str], int] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _recovery_modes: dict[tuple[str, str], str] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _universal_primer_evidence: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _target_specific_recovery: dict[tuple[str, str], list[tuple[str, str, str]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def apply(self, row: dict, *, row_index: int) -> Record:
        try:
            review = ThreeWayJunctionReviewV1.model_validate(row)
        except ValidationError as exc:
            detail = format_validation_error(exc)
            raise SchemaError(f"Invalid three_way_junction_review_v1 contract at row {row_index}: {detail}") from None

        target_identity = (review.source.plan_id, review.target.target_id)
        if target_identity in self._target_identities:
            raise SchemaError(f"three_way_junction_review_v1 document has duplicate target identity at row {row_index}")

        source_receipt = review.source.model_dump(mode="json")
        previous_source = self._source_receipts.get(review.source.plan_id)
        if previous_source is not None and previous_source != source_receipt:
            raise SchemaError(
                f"three_way_junction_review_v1 document has contradictory source metadata at row {row_index}"
            )
        self._source_receipts[review.source.plan_id] = source_receipt

        if review.source.algorithm == JUNCTION_STRING_V1_ALGORITHM:
            complement_end_preparation = review.geometry.junctions[0].complement_end_preparation
            previous_preparation = self._v1_complement_end_preparation.get(review.source.plan_id)
            if previous_preparation is not None and previous_preparation != complement_end_preparation:
                raise SchemaError(
                    "three_way_junction_review_v1 document has contradictory plan-wide "
                    f"complement-end preparation at row {row_index}"
                )
            self._v1_complement_end_preparation[review.source.plan_id] = complement_end_preparation

        plan_and_group = (review.source.plan_id, review.search.assembly_group_id)
        search_receipt = review.search.model_dump(mode="json")
        previous_receipt = self._search_receipts.get(plan_and_group)
        if previous_receipt is not None and previous_receipt != search_receipt:
            raise SchemaError(
                "three_way_junction_review_v1 document has a contradictory "
                f"assembly-group-wide search receipt at row {row_index}"
            )
        self._search_receipts[plan_and_group] = search_receipt

        assembly_group_check_evidence = tuple(
            sorted(
                (
                    check.subject.kind,
                    check.subject.id,
                    check.check,
                    check.status,
                    check.detail,
                )
                for check in review.checks
                if check.subject.kind == "assembly_group"
            )
        )
        previous_check_evidence = self._assembly_group_check_evidence.get(plan_and_group)
        if previous_check_evidence is not None and previous_check_evidence != assembly_group_check_evidence:
            raise SchemaError(
                "three_way_junction_review_v1 document has contradictory "
                f"assembly-group-wide check evidence at row {row_index}"
            )
        self._assembly_group_check_evidence[plan_and_group] = assembly_group_check_evidence

        grouped_sequence_identity = (*plan_and_group, review.target.sequence_sha256)
        if grouped_sequence_identity in self._grouped_sequence_identities:
            raise SchemaError(
                "three_way_junction_review_v1 document has a duplicate target sequence in one assembly group"
            )

        fragment_identities = {(review.source.plan_id, fragment.fragment_id) for fragment in review.geometry.fragments}
        if fragment_identities & self._fragment_identities:
            raise SchemaError("three_way_junction_review_v1 document has duplicate plan-scoped fragment identity")

        junction_identities = {(review.source.plan_id, junction.junction_id) for junction in review.geometry.junctions}
        if junction_identities & self._junction_identities:
            raise SchemaError("three_way_junction_review_v1 document has duplicate plan-scoped junction identity")

        barcode_identities = {
            (*plan_and_group, min(junction.barcode, junction.barcode_complement))
            for junction in review.geometry.junctions
        }
        if len(barcode_identities) != len(review.geometry.junctions):
            raise SchemaError(
                "three_way_junction_review_v1 document has duplicate barcode identity in one assembly group"
            )
        if barcode_identities & self._barcode_identities:
            raise SchemaError(
                "three_way_junction_review_v1 document has duplicate barcode identity in one assembly group"
            )

        previous_mode = self._recovery_modes.get(plan_and_group)
        if previous_mode is not None and previous_mode != review.recovery.mode:
            raise SchemaError("three_way_junction_review_v1 document mixes recovery modes within one assembly group")

        if review.recovery.mode == "universal":
            primer_evidence = (
                review.recovery.forward.model_dump(mode="json", exclude={"target_binding_span"}),
                review.recovery.reverse.model_dump(mode="json", exclude={"target_binding_span"}),
            )
            previous_primers = self._universal_primer_evidence.get(plan_and_group)
            if previous_primers is not None and previous_primers != primer_evidence:
                raise SchemaError(
                    "three_way_junction_review_v1 document has contradictory universal recovery primer evidence"
                )
            self._universal_primer_evidence[plan_and_group] = primer_evidence
        else:
            reverse_span = review.recovery.reverse.target_binding_span
            reverse_target_suffix = review.target.sequence_5to3[reverse_span.start : reverse_span.end]
            self._target_specific_recovery.setdefault(plan_and_group, []).append(
                (
                    review.target.sequence_5to3,
                    review.recovery.forward.binding_sequence_5to3,
                    reverse_target_suffix,
                )
            )

        self._target_identities.add(target_identity)
        next_junction_count = self._junction_counts.get(plan_and_group, 0) + len(review.geometry.junctions)
        if next_junction_count > review.search.locus_count:
            raise SchemaError("three_way_junction_review_v1 document junction total exceeds declared locus_count")

        self._grouped_sequence_identities.add(grouped_sequence_identity)
        self._fragment_identities.update(fragment_identities)
        self._junction_identities.update(junction_identities)
        self._barcode_identities.update(barcode_identities)
        self._recovery_modes[plan_and_group] = review.recovery.mode
        self._junction_counts[plan_and_group] = next_junction_count

        record = Record(
            id=review.target.target_id,
            alphabet=self.alphabet,
            sequence=review.target.sequence_5to3,
            features=(),
            effects=(),
            display=Display(overlay_text=None, tag_labels={}),
            meta={
                "adapter": "three_way_junction_review_v1",
                "topology_kind": "fragment_pool",
                "three_way_junction_review": review.model_dump(mode="json"),
            },
        )
        try:
            return record.validate()
        except ContractError as exc:
            raise SchemaError(str(exc)) from exc

    def finalize(self) -> None:
        if not self._junction_counts:
            raise SchemaError("three_way_junction_review_v1 document must contain at least one target")
        for recovery_rows in self._target_specific_recovery.values():
            for target_index, (_, forward_binding, reverse_target_suffix) in enumerate(recovery_rows):
                if any(
                    other_index != target_index
                    and other_sequence.startswith(forward_binding)
                    and other_sequence.endswith(reverse_target_suffix)
                    for other_index, (other_sequence, _, _) in enumerate(recovery_rows)
                ):
                    raise SchemaError(
                        "three_way_junction_review_v1 document has ambiguous target-specific recovery "
                        "within one assembly group"
                    )
        for plan_and_group, junction_count in self._junction_counts.items():
            declared_locus_count = self._search_receipts[plan_and_group]["locus_count"]
            if junction_count != declared_locus_count:
                raise SchemaError(
                    "three_way_junction_review_v1 document junction total does not match declared locus_count"
                )


__all__ = ["ThreeWayJunctionReviewV1Adapter"]
