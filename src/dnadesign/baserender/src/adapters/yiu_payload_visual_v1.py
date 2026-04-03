"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/adapters/yiu_payload_visual_v1.py

Adapter from YIU payload visual contracts to baserender Record v1.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from dnadesign.contracts.visual import YiuPayloadVisualV1

from ..core import ContractError, Record, SchemaError, Span
from ..core.record import Display, Effect, Feature
from .sequence_evidence_map_v1 import SequenceEvidenceMapV1Adapter


def _junction_span(contract: YiuPayloadVisualV1) -> dict[str, int]:
    return {"start": contract.junction.start, "end": contract.junction.end}


def _build_base_contract(contract: YiuPayloadVisualV1) -> dict[str, object]:
    mismatch_indices = [entry.payload_index for entry in contract.mismatches]
    junction_hidden = [
        index for index in range(contract.junction.start, contract.junction.end) if index not in set(mismatch_indices)
    ]
    row_labels_raw = contract.meta.get("row_labels")
    row_labels = row_labels_raw if isinstance(row_labels_raw, Mapping) else {}
    payload_label = str(row_labels.get("primary") or "Selected payload")
    complement_label = str(row_labels.get("complement") or "Selected complement")
    return {
        "contract_kind": "sequence_evidence_map_v1",
        "state_id": contract.state_id,
        "topology_kind": "linear_dsdna",
        "alphabet": contract.alphabet,
        "primary_sequence": contract.selected_payload_sequence,
        "complement_sequence": contract.selected_complement_sequence,
        "owners": [],
        "effect_tags": [],
        "boundaries": [
            {
                "boundary_id": "junction_start",
                "row_id": "primary",
                "boundary": contract.junction.start,
                "boundary_kind": "ligation_junction",
                "display_label": "Junction start",
                "short_label": "J0",
            },
            {
                "boundary_id": "junction_end",
                "row_id": "primary",
                "boundary": contract.junction.end,
                "boundary_kind": "ligation_junction",
                "display_label": "Junction end",
                "short_label": "J4",
            },
        ],
        "pairings": [],
        "display": {"title": contract.display.title},
        "meta": {
            "row_labels": {
                "primary": payload_label,
                "complement": complement_label,
            },
            "base_highlights": {
                "primary": mismatch_indices,
                "complement": mismatch_indices,
            },
            "connector_hidden_indices": junction_hidden,
            "connector_cross_indices": mismatch_indices,
            "connector_overhang_spans": [_junction_span(contract)],
            "segment_labels": [
                {"text": "Left body", "start": 0, "end": contract.junction.start},
                {
                    "text": "Right body",
                    "start": contract.junction.end,
                    "end": len(contract.selected_payload_sequence),
                },
            ]
            if contract.junction.start > 0 and contract.junction.end < len(contract.selected_payload_sequence)
            else [],
            "reference_payload_sequence": contract.reference_payload_sequence,
            "show_reference_payload_row": contract.show_reference_payload_row,
            "yiu_payload_meta": dict(contract.meta),
        },
    }


def _motif_track(reference_strand: str) -> int:
    return 0 if reference_strand == "+" else 1


@dataclass(frozen=True)
class YiuPayloadVisualV1Adapter:
    columns: Mapping[str, Any]
    policies: Mapping[str, Any]
    alphabet: str

    def apply(self, row: dict, *, row_index: int) -> Record:
        try:
            contract = YiuPayloadVisualV1.model_validate(row)
        except Exception as exc:
            raise SchemaError(f"Invalid yiu_payload_visual_v1 contract at row {row_index}: {exc}") from exc

        base_adapter = SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet=self.alphabet)
        base_record = base_adapter.apply(_build_base_contract(contract), row_index=row_index)

        features = list(base_record.features)
        effects = list(base_record.effects)
        tag_labels = dict(base_record.display.tag_labels)
        for motif in contract.motif_layers:
            feature_id = f"motif:{motif.motif_instance_id}"
            track = _motif_track(motif.reference_strand)
            tag_labels.setdefault(f"tf:{motif.tf_name}", motif.tf_name)
            features.append(
                Feature(
                    id=feature_id,
                    kind="regulator_window",
                    span=Span(
                        start=motif.start,
                        end=motif.end,
                        strand="fwd" if motif.reference_strand == "+" else "rev",
                    ),
                    label=motif.label,
                    tags=(f"tf:{motif.tf_name}",),
                    attrs={
                        "tf": motif.tf_name,
                        "motif_name": motif.motif_name,
                        "lane": "primary" if motif.reference_strand == "+" else "complement",
                    },
                    render={"track": track, "priority": 10},
                )
            )
            effects.append(
                Effect(
                    kind="motif_logo",
                    target={"feature_id": feature_id},
                    params={"matrix": motif.matrix},
                    render={"track": track, "priority": 20},
                )
            )

        record = Record(
            id=base_record.id,
            alphabet=base_record.alphabet,
            sequence=base_record.sequence,
            features=tuple(features),
            effects=tuple(effects),
            display=Display(
                overlay_text=base_record.display.overlay_text,
                video_subtitle=base_record.display.video_subtitle,
                tag_labels=tag_labels,
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
