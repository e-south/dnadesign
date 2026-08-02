"""Adapter from neutral three-way-junction review evidence to Record v1."""

from __future__ import annotations

from dataclasses import dataclass
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

    def apply(self, row: dict, *, row_index: int) -> Record:
        try:
            review = ThreeWayJunctionReviewV1.model_validate(row)
        except ValidationError as exc:
            detail = format_validation_error(exc)
            raise SchemaError(f"Invalid three_way_junction_review_v1 contract at row {row_index}: {detail}") from None

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
