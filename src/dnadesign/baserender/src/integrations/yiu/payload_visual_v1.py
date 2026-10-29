"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/yiu/payload_visual_v1.py

Adapter from YIU payload visual contracts to baserender Record v1.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from dnadesign.contracts.visual import YiuPayloadVisualV1

from ...core import ContractError, Record, SchemaError
from ...core.record import Display
from ..generic.sequence_evidence_map_v1 import SequenceEvidenceMapV1Adapter
from .payload_motif_overlay import build_motif_overlay
from .payload_sequence_projection import build_sequence_evidence_map_contract


@dataclass(frozen=True)
class YiuPayloadVisualV1Adapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str

    def _build_base_record(self, *, contract: YiuPayloadVisualV1, row_index: int) -> Record:
        base_adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet=self.alphabet)
        return base_adapter.apply(build_sequence_evidence_map_contract(contract), row_index=row_index)

    @staticmethod
    def _merge_tag_labels(*, base_record: Record, overlay_tag_labels: dict[str, str]) -> dict[str, str]:
        tag_labels = dict(base_record.display.tag_labels)
        for tag, label in overlay_tag_labels.items():
            tag_labels.setdefault(tag, label)
        return tag_labels

    def apply(self, row: dict, *, row_index: int) -> Record:
        try:
            contract = YiuPayloadVisualV1.model_validate(row)
        except Exception as exc:
            raise SchemaError(f"Invalid yiu_payload_visual_v1 contract at row {row_index}: {exc}") from exc

        base_record = self._build_base_record(contract=contract, row_index=row_index)
        overlay = build_motif_overlay(contract, base_record=base_record)

        record = Record(
            id=base_record.id,
            alphabet=base_record.alphabet,
            sequence=base_record.sequence,
            features=base_record.features + overlay.features,
            effects=base_record.effects + overlay.effects,
            display=Display(
                overlay_text=base_record.display.overlay_text,
                video_subtitle=base_record.display.video_subtitle,
                tag_labels=self._merge_tag_labels(base_record=base_record, overlay_tag_labels=overlay.tag_labels),
                trajectory_panel=base_record.display.trajectory_panel,
            ),
            meta={
                **dict(base_record.meta),
                "adapter": "yiu_payload_visual_v1",
                "contract": contract.model_dump(mode="json"),
                "reference_payload_sequence": contract.reference_payload_sequence,
                "show_reference_payload_row": contract.show_reference_payload_row,
                "view_meta": dict(contract.meta),
            },
        )
        try:
            return record.validate()
        except ContractError as exc:
            raise SchemaError(str(exc)) from exc
