"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/adapters/yiu_payload_sequence_projection.py

Sequence-evidence projection helpers for YIU payload visual contracts.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Mapping

from dnadesign.contracts.visual import YiuPayloadVisualV1


def _junction_span(contract: YiuPayloadVisualV1) -> dict[str, int]:
    return {"start": contract.junction.start, "end": contract.junction.end}


def _normalized_row_labels(meta: Mapping[str, object]) -> dict[str, str]:
    row_labels_raw = meta.get("row_labels")
    row_labels = row_labels_raw if isinstance(row_labels_raw, Mapping) else {}
    return {
        "primary": str(row_labels.get("primary") or "").strip(),
        "complement": str(row_labels.get("complement") or "").strip(),
    }


def build_sequence_evidence_map_contract(contract: YiuPayloadVisualV1) -> dict[str, object]:
    mismatch_indices = [entry.payload_index for entry in contract.mismatches]
    mismatch_index_set = set(mismatch_indices)
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
                "short_label": "",
            },
            {
                "boundary_id": "junction_end",
                "row_id": "complement",
                "boundary": contract.junction.end,
                "boundary_kind": "ligation_junction",
                "display_label": "Junction end",
                "short_label": "",
            },
        ],
        "pairings": [],
        "display": {"title": contract.display.title},
        "meta": {
            "row_labels": _normalized_row_labels(contract.meta),
            "base_highlights": {
                "primary": mismatch_indices,
                "complement": mismatch_indices,
            },
            "connector_hidden_indices": [
                index
                for index in range(contract.junction.start, contract.junction.end)
                if index not in mismatch_index_set
            ],
            "connector_cross_indices": mismatch_indices,
            "connector_overhang_spans": [_junction_span(contract)],
            "reference_payload_sequence": contract.reference_payload_sequence,
            "show_reference_payload_row": contract.show_reference_payload_row,
            "yiu_payload_meta": dict(contract.meta),
        },
    }


__all__ = ["build_sequence_evidence_map_contract"]
