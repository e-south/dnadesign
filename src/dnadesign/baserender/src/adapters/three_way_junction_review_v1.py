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
    _target_identities: set[tuple[str, str]] = field(
        default_factory=set,
        init=False,
        repr=False,
        compare=False,
    )
    _physical_sequence_identities: set[tuple[str, str, str]] = field(
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

        plan_and_pool = (review.source.plan_id, review.search.pool_id)
        search_receipt = review.search.model_dump(mode="json")
        previous_receipt = self._search_receipts.get(plan_and_pool)
        if previous_receipt is not None and previous_receipt != search_receipt:
            raise SchemaError(
                f"three_way_junction_review_v1 document has a contradictory pool-wide search receipt at row {row_index}"
            )
        self._search_receipts[plan_and_pool] = search_receipt

        physical_sequence_identity = (*plan_and_pool, review.target.sequence_sha256)
        if physical_sequence_identity in self._physical_sequence_identities:
            raise SchemaError("three_way_junction_review_v1 document has duplicate physical target sequence")

        previous_mode = self._recovery_modes.get(plan_and_pool)
        if previous_mode is not None and previous_mode != review.recovery.mode:
            raise SchemaError("three_way_junction_review_v1 document mixes recovery modes within one physical pool")

        if review.recovery.mode == "universal":
            primer_evidence = (
                review.recovery.forward.model_dump(mode="json", exclude={"target_binding_span"}),
                review.recovery.reverse.model_dump(mode="json", exclude={"target_binding_span"}),
            )
            previous_primers = self._universal_primer_evidence.get(plan_and_pool)
            if previous_primers is not None and previous_primers != primer_evidence:
                raise SchemaError(
                    "three_way_junction_review_v1 document has contradictory universal recovery primer evidence"
                )
            self._universal_primer_evidence[plan_and_pool] = primer_evidence

        self._target_identities.add(target_identity)
        next_junction_count = self._junction_counts.get(plan_and_pool, 0) + len(review.geometry.junctions)
        if next_junction_count > review.search.locus_count:
            raise SchemaError("three_way_junction_review_v1 document junction total exceeds declared locus_count")

        self._physical_sequence_identities.add(physical_sequence_identity)
        self._recovery_modes[plan_and_pool] = review.recovery.mode
        self._junction_counts[plan_and_pool] = next_junction_count

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
        for plan_and_pool, junction_count in self._junction_counts.items():
            declared_locus_count = self._search_receipts[plan_and_pool]["locus_count"]
            if junction_count != declared_locus_count:
                raise SchemaError(
                    "three_way_junction_review_v1 document junction total does not match declared locus_count"
                )


__all__ = ["ThreeWayJunctionReviewV1Adapter"]
