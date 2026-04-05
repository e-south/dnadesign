"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/adapters/yiu_payload_sequence_projection.py

Sequence-evidence projection helpers for YIU payload visual contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.contracts.visual import YiuPayloadVisualV1
from dnadesign.contracts.visual.sequence_evidence_meta import (
    build_sequence_evidence_connector_meta,
    normalize_sequence_evidence_row_labels,
)


def _build_junction_boundaries(contract: YiuPayloadVisualV1) -> list[dict[str, object]]:
    return [
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
    ]


def _build_projection_meta(contract: YiuPayloadVisualV1, *, mismatch_indices: list[int]) -> dict[str, object]:
    return {
        "row_labels": normalize_sequence_evidence_row_labels(contract.meta),
        "base_highlights": {
            "primary": mismatch_indices,
            "complement": mismatch_indices,
        },
        "reference_payload_sequence": contract.reference_payload_sequence,
        "show_reference_payload_row": contract.show_reference_payload_row,
        **build_sequence_evidence_connector_meta(
            start=contract.junction.start,
            end=contract.junction.end,
            cross_indices=mismatch_indices,
        ),
    }


def build_sequence_evidence_map_contract(contract: YiuPayloadVisualV1) -> dict[str, object]:
    mismatch_indices = [entry.payload_index for entry in contract.mismatches]
    return {
        "contract_kind": "sequence_evidence_map_v1",
        "state_id": contract.state_id,
        "topology_kind": "linear_dsdna",
        "alphabet": contract.alphabet,
        "primary_sequence": contract.selected_payload_sequence,
        "complement_sequence": contract.selected_complement_sequence,
        "owners": [],
        "effect_tags": [],
        "boundaries": _build_junction_boundaries(contract),
        "pairings": [],
        "display": {"title": contract.display.title},
        "meta": _build_projection_meta(contract, mismatch_indices=mismatch_indices),
    }


__all__ = ["build_sequence_evidence_map_contract"]
