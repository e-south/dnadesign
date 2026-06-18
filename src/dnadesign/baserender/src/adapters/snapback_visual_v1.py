"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/adapters/snapback_visual_v1.py

Adapter from shared snapback visual contracts to baserender Record v1.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from dnadesign.contracts.visual import SnapbackVisualV1

from ..core import ContractError, Record, SchemaError
from ..core.record import Display


@dataclass(frozen=True)
class SnapbackVisualV1Adapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str

    def apply(self, row: dict, *, row_index: int) -> Record:
        try:
            contract = SnapbackVisualV1.model_validate(row)
        except Exception as exc:
            raise SchemaError(f"Invalid snapback_visual_v1 contract at row {row_index}: {exc}") from exc

        record = Record(
            id=contract.state_id,
            alphabet=self.alphabet,
            sequence=contract.primary_sequence,
            features=(),
            effects=(),
            display=Display(overlay_text=None, tag_labels={}),
            meta={
                "adapter": "snapback_visual_v1",
                "contract": contract.model_dump(mode="json"),
                "view_meta": dict(contract.meta),
            },
        )
        try:
            return record.validate()
        except ContractError as exc:
            raise SchemaError(str(exc)) from exc
