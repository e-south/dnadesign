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
    build_sequence_evidence_connector_span_meta,
    build_sequence_evidence_span_backdrop_meta,
    normalize_sequence_evidence_row_labels,
)
from dnadesign.contracts.visual.yiu_payload_visual_v1 import YiuPayloadMismatchV1

YIU_MISMATCH_HIGHLIGHT_COLOR = "#B91C1C"


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


def _build_base_highlights(mismatches: list[YiuPayloadMismatchV1]) -> dict[str, list[int]]:
    highlights = {"primary": [], "complement": []}
    for mismatch in mismatches:
        row_id = "primary" if mismatch.mutated_strand == "payload" else "complement"
        highlights[row_id].append(mismatch.payload_index)
    for row_id in highlights:
        highlights[row_id].sort()
    return highlights


def _build_projection_meta(contract: YiuPayloadVisualV1) -> dict[str, object]:
    backdrop_meta = contract.meta.get("span_backdrops")
    return {
        "row_labels": normalize_sequence_evidence_row_labels(contract.meta),
        "base_highlights": _build_base_highlights(contract.mismatches),
        "base_highlight_color": YIU_MISMATCH_HIGHLIGHT_COLOR,
        "reference_payload_sequence": contract.reference_payload_sequence,
        "show_reference_payload_row": contract.show_reference_payload_row,
        **(
            {"span_backdrops": backdrop_meta}
            if backdrop_meta is not None
            else build_sequence_evidence_span_backdrop_meta(
                start=contract.junction.start,
                end=contract.junction.end,
                coordinate_space="payload_forward",
            )
        ),
        **build_sequence_evidence_connector_span_meta(
            start=contract.junction.start,
            end=contract.junction.end,
        ),
    }


def build_sequence_evidence_map_contract(contract: YiuPayloadVisualV1) -> dict[str, object]:
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
        "meta": _build_projection_meta(contract),
    }


__all__ = ["build_sequence_evidence_map_contract"]
