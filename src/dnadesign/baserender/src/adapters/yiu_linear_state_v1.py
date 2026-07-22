"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/adapters/yiu_linear_state_v1.py

Adapter from YIU linear-state contracts to baserender Record v1.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from dnadesign.contracts.visual import YiuLinearStateV1

from ..core import ContractError, Record, SchemaError, Span
from ..core.record import Display, Effect, Feature


def _require_mapping(value: object, *, ctx: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{ctx} must be a mapping")
    return value


@dataclass(frozen=True)
class YiuLinearStateV1Adapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str

    def apply(self, row: dict, *, row_index: int) -> Record:
        payload = _require_mapping(row, ctx=f"row {row_index}")
        try:
            contract = YiuLinearStateV1.model_validate(payload)
        except Exception as exc:
            raise SchemaError(f"Invalid yiu_linear_state_v1 contract at row {row_index}: {exc}") from exc
        sequence = contract.primary_sequence

        features: list[Feature] = []
        for segment in contract.segments:
            if not isinstance(segment, Mapping):
                raise SchemaError("yiu_linear_state_v1.segments entries must be mappings")
            start = int(segment.get("state_start", 0))
            end = int(segment.get("state_end", 0))
            if end <= start:
                continue
            segment_id = str(segment.get("segment_id") or "segment")
            features.append(
                Feature(
                    id=segment_id,
                    kind="interval_annotation",
                    span=Span(start=start, end=end, strand="fwd"),
                    label=segment_id,
                    tags=("yiu:segment",),
                    attrs={
                        "lane": "primary",
                        "shape": "band",
                        "semantic": segment_id,
                        "intent": "structural",
                        "style_token": "segment",
                    },
                    render={},
                )
            )

        effects: list[Effect] = []
        for cut in contract.cuts:
            if not isinstance(cut, Mapping):
                continue
            for boundary_key in ("top_boundary", "bottom_boundary"):
                boundary = cut.get(boundary_key)
                if boundary is None:
                    continue
                effects.append(
                    Effect(
                        kind="boundary_marker",
                        target={"boundary": int(boundary), "lane": "primary"},
                        params={
                            "label": str(cut.get("site_id") or boundary_key),
                            "semantic": "cut_boundary",
                            "intent": "structural",
                        },
                        render={},
                    )
                )
        for junction in contract.junctions:
            if not isinstance(junction, Mapping):
                continue
            join_index = junction.get("join_index")
            if join_index is None:
                continue
            effects.append(
                Effect(
                    kind="boundary_marker",
                    target={"boundary": int(join_index), "lane": "primary"},
                    params={
                        "label": str(junction.get("id") or "junction"),
                        "semantic": "junction_boundary",
                        "intent": "structural",
                    },
                    render={},
                )
            )

        record = Record(
            id=contract.state_id or f"row_{row_index}",
            alphabet=self.alphabet,
            sequence=sequence,
            features=tuple(features),
            effects=tuple(effects),
            display=Display(overlay_text=str(contract.display.title or "")),
            meta={
                "adapter": "yiu_linear_state_v1",
                "contract": contract.model_dump(mode="json"),
                "complement_sequence": contract.complement_sequence,
                "view_meta": dict(contract.meta),
            },
        )
        try:
            return record.validate()
        except ContractError as exc:
            raise SchemaError(str(exc)) from exc
