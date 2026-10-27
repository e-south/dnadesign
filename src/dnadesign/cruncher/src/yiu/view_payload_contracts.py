"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/yiu/view_payload_contracts.py

Payload-view contract shells for payload-centric YIU bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.contracts.visual import YiuPayloadVisualV1
from dnadesign.contracts.visual.yiu_payload_visual_v1 import (
    YiuPayloadDisplayV1,
    YiuPayloadJunctionV1,
)
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload
from dnadesign.cruncher.yiu.view_payload_content import (
    build_payload_mismatch_annotations,
    build_payload_motif_layers,
    build_payload_view_meta,
)
from dnadesign.cruncher.yiu.view_styles import build_payload_view_title


def build_payload_view_contract(normalized: NormalizedPayload) -> dict[str, object]:
    return YiuPayloadVisualV1(
        state_id="payload",
        alphabet="iupac_dna",
        reference_payload_sequence=normalized.reference_payload_sequence,
        selected_payload_sequence=normalized.selected_payload_sequence,
        selected_complement_sequence=normalized.selected_complement_sequence,
        show_reference_payload_row=normalized.selected_payload_sequence != normalized.reference_payload_sequence,
        junction=YiuPayloadJunctionV1(
            start=normalized.junction.start,
            end=normalized.junction.end,
            offsets=[0, 1, 2, 3],
        ),
        mismatches=build_payload_mismatch_annotations(normalized),
        motif_layers=build_payload_motif_layers(normalized),
        display=YiuPayloadDisplayV1(title=build_payload_view_title(normalized)),
        meta=build_payload_view_meta(normalized),
    ).model_dump(mode="json")


__all__ = ["build_payload_view_contract"]
