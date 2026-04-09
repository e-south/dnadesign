"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/adapters/yiu_topology_cartoon_v1.py

Adapter from YIU topology-cartoon contracts to baserender Record v1.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from dnadesign.contracts.visual import YiuTopologyCartoonV1

from ..core import ContractError, Record, SchemaError, Span
from ..core.record import Display, Feature


def _require_mapping(value: object, *, ctx: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{ctx} must be a mapping")
    return value


@dataclass(frozen=True)
class YiuTopologyCartoonV1Adapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str

    def apply(self, row: dict, *, row_index: int) -> Record:
        payload = _require_mapping(row, ctx=f"row {row_index}")
        try:
            contract = YiuTopologyCartoonV1.model_validate(payload)
        except Exception as exc:
            raise SchemaError(f"Invalid yiu_topology_cartoon_v1 contract at row {row_index}: {exc}") from exc
        sequence = str(contract.sequence or "N")
        features: list[Feature] = []
        for segment in contract.segments:
            if not isinstance(segment, Mapping):
                continue
            start = int(segment.get("state_start", 0))
            end = int(segment.get("state_end", 0))
            if end <= start:
                continue
            features.append(
                Feature(
                    id=str(segment.get("segment_id") or "segment"),
                    kind="interval_annotation",
                    span=Span(start=start, end=end, strand="fwd"),
                    label=str(segment.get("segment_id") or "segment"),
                    tags=("yiu:topology_segment",),
                    attrs={
                        "lane": "topology",
                        "shape": "band",
                        "semantic": str(segment.get("segment_id") or "segment"),
                        "intent": "structural",
                        "style_token": "segment",
                    },
                    render={},
                )
            )
        record = Record(
            id=contract.state_id or f"row_{row_index}",
            alphabet=self.alphabet,
            sequence=sequence,
            features=tuple(features),
            display=Display(overlay_text=str(contract.display.title or "")),
            meta={
                "adapter": "yiu_topology_cartoon_v1",
                "topology_cartoon": contract.model_dump(mode="json"),
                "view_meta": dict(contract.meta),
            },
        )
        try:
            return record.validate()
        except ContractError as exc:
            raise SchemaError(str(exc)) from exc
