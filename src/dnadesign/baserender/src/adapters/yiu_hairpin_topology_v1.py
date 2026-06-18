"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/adapters/yiu_hairpin_topology_v1.py

Adapter from YIU hairpin-topology contracts to baserender Record v1.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from dnadesign.contracts.visual import YiuHairpinTopologyV1

from ..core import ContractError, Record, SchemaError, Span
from ..core.record import Display, Effect, Feature


def _require_mapping(value: object, *, ctx: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{ctx} must be a mapping")
    return value


def _span(raw: object, *, ctx: str) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise SchemaError(f"{ctx} must be a 2-item list")
    start = int(raw[0])
    end = int(raw[1])
    if end <= start:
        raise SchemaError(f"{ctx} end must be > start")
    return start, end


@dataclass(frozen=True)
class YiuHairpinTopologyV1Adapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str

    def apply(self, row: dict, *, row_index: int) -> Record:
        payload = _require_mapping(row, ctx=f"row {row_index}")
        try:
            contract = YiuHairpinTopologyV1.model_validate(payload)
        except Exception as exc:
            raise SchemaError(f"Invalid yiu_hairpin_topology_v1 contract at row {row_index}: {exc}") from exc
        sequence = contract.sequence

        stem_left = (contract.stem_left_span.start, contract.stem_left_span.end)
        stem_right = (contract.stem_right_span.start, contract.stem_right_span.end)
        loop = (contract.loop_span.start, contract.loop_span.end)
        features = [
            Feature(
                id="stem5p_span",
                kind="interval_annotation",
                span=Span(start=stem_left[0], end=stem_left[1], strand="fwd"),
                label="Stem 5' arm",
                tags=("topology:stem5p_arm",),
                attrs={"lane": "topology", "shape": "band", "semantic": "stem5p_arm", "style_token": "segment_stem5p"},
                render={},
            ),
            Feature(
                id="loop_span",
                kind="interval_annotation",
                span=Span(start=loop[0], end=loop[1], strand="fwd"),
                label="Loop",
                tags=("topology:loop",),
                attrs={"lane": "topology", "shape": "band", "semantic": "loop", "style_token": "segment_loop"},
                render={},
            ),
            Feature(
                id="stem3p_span",
                kind="interval_annotation",
                span=Span(start=stem_right[0], end=stem_right[1], strand="fwd"),
                label="Stem 3' arm",
                tags=("topology:stem3p_arm",),
                attrs={"lane": "topology", "shape": "band", "semantic": "stem3p_arm", "style_token": "segment_stem3p"},
                render={},
            ),
        ]
        record = Record(
            id=contract.state_id or f"row_{row_index}",
            alphabet=self.alphabet,
            sequence=sequence,
            features=tuple(features),
            effects=(
                Effect(
                    kind="pair_map",
                    target={"pairs": [pair.model_dump(mode="json") for pair in contract.pair_map]},
                    params={"semantic": "stem_pairing"},
                    render={},
                ),
            ),
            display=Display(overlay_text=str(contract.display.title or "")),
            meta={
                "adapter": "yiu_hairpin_topology_v1",
                "hairpin_topology": {
                    "stem5p_span": {"start": stem_left[0], "end": stem_left[1]},
                    "loop_span": {"start": loop[0], "end": loop[1]},
                    "stem3p_span": {"start": stem_right[0], "end": stem_right[1]},
                },
                "hairpin_notes": [dict(item) for item in contract.annotations if isinstance(item, Mapping)],
                "view_meta": dict(contract.meta),
            },
        )
        try:
            return record.validate()
        except ContractError as exc:
            raise SchemaError(str(exc)) from exc
