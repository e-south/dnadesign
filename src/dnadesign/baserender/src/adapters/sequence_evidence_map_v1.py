"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/adapters/sequence_evidence_map_v1.py

Adapter from shared sequence-evidence contracts to baserender Record v1.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from dnadesign.contracts.visual import SequenceEvidenceMapV1

from ..core import ContractError, Record, SchemaError, Span
from ..core.record import Display, Effect, Feature


def _strand(row_id: str) -> str:
    return "fwd" if row_id == "primary" else "rev"


def _style_token_for_owner(owner_id: str) -> str:
    if "payload" in owner_id:
        return "segment_payload"
    if "adapter" in owner_id:
        return "segment_adapter"
    if "primer" in owner_id:
        return "segment_primer"
    if "retained" in owner_id:
        return "segment_retained"
    if "sacrificial" in owner_id:
        return "segment_sacrificial"
    return "segment"


def _style_token_for_tag(tag_kind: str) -> str:
    if "overhang" in tag_kind:
        return "site_overhang"
    if "recognition" in tag_kind:
        return "site_recognition"
    if "primer" in tag_kind:
        return "site_primer"
    if "adapter" in tag_kind:
        return "site_adapter"
    if "boundary" in tag_kind or "junction" in tag_kind:
        return "site_boundary"
    return "site_effect"


@dataclass(frozen=True)
class SequenceEvidenceMapV1Adapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str

    def apply(self, row: dict, *, row_index: int) -> Record:
        try:
            contract = SequenceEvidenceMapV1.model_validate(row)
        except Exception as exc:
            raise SchemaError(f"Invalid sequence_evidence_map_v1 contract at row {row_index}: {exc}") from exc

        features: list[Feature] = []
        tag_labels: dict[str, str] = {}

        for owner in contract.owners:
            tag = f"owner:{owner.owner_id}"
            tag_labels[tag] = owner.display_label
            features.append(
                Feature(
                    id=f"{owner.row_id}:{owner.owner_id}:{owner.start}:{owner.end}",
                    kind="interval_annotation",
                    span=Span(start=owner.start, end=owner.end, strand=_strand(owner.row_id)),
                    label=owner.short_label,
                    tags=(tag,),
                    attrs={
                        "lane": owner.row_id,
                        "shape": "band",
                        "semantic": owner.owner_id,
                        "intent": "owner",
                        "style_token": _style_token_for_owner(owner.owner_id),
                    },
                    render={"track": 0 if owner.row_id == "primary" else 1},
                )
            )

        for tag in contract.effect_tags:
            feature_tag = f"effect:{tag.tag_kind}"
            tag_labels[feature_tag] = tag.display_label
            features.append(
                Feature(
                    id=f"{tag.row_id}:{tag.tag_id}:{tag.start}:{tag.end}",
                    kind="interval_annotation",
                    span=Span(start=tag.start, end=tag.end, strand=_strand(tag.row_id)),
                    label=tag.short_label,
                    tags=(feature_tag,),
                    attrs={
                        "lane": tag.row_id,
                        "shape": "rounded_rect",
                        "semantic": tag.tag_kind,
                        "intent": "effect",
                        "style_token": _style_token_for_tag(tag.tag_kind),
                    },
                    render={},
                )
            )

        effects: list[Effect] = []
        for boundary in contract.boundaries:
            effects.append(
                Effect(
                    kind="boundary_marker",
                    target={"boundary": boundary.boundary, "lane": boundary.row_id},
                    params={
                        "label": boundary.short_label,
                        "semantic": boundary.boundary_kind,
                        "intent": "evidence_boundary",
                    },
                    render={},
                )
            )
        for pairing in contract.pairings:
            effects.append(
                Effect(
                    kind="span_link",
                    target={
                        "from_span": {"start": pairing.primary_start, "end": pairing.primary_end, "strand": "fwd"},
                        "to_span": {
                            "start": pairing.complement_start,
                            "end": pairing.complement_end,
                            "strand": "rev",
                        },
                    },
                    params={"label": pairing.short_label or pairing.display_label or "", "lane": "top"},
                    render={"track": 4},
                )
            )

        record = Record(
            id=contract.state_id,
            alphabet=self.alphabet,
            sequence=contract.primary_sequence,
            features=tuple(features),
            effects=tuple(effects),
            display=Display(overlay_text=str(contract.display.title or ""), tag_labels=tag_labels),
            meta={
                "adapter": "sequence_evidence_map_v1",
                "contract": contract.model_dump(mode="json"),
                "complement_sequence": contract.complement_sequence,
                "view_meta": dict(contract.meta),
                "row_labels": {"primary": "Primary", "complement": "Complement"},
                "show_reverse_complement": contract.complement_sequence is not None,
            },
        )
        try:
            return record.validate()
        except ContractError as exc:
            raise SchemaError(str(exc)) from exc
