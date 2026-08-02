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

    def apply(self, row: dict, *, row_index: int) -> Record:
        try:
            review = ThreeWayJunctionReviewV1.model_validate(row)
        except ValidationError as exc:
            detail = format_validation_error(exc)
            raise SchemaError(f"Invalid three_way_junction_review_v1 contract at row {row_index}: {detail}") from None

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


__all__ = ["ThreeWayJunctionReviewV1Adapter"]
