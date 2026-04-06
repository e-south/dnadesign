"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/view_payload_content.py

Payload-view content policy for payload-centric YIU bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.contracts.visual.sequence_evidence_meta import build_sequence_evidence_span_backdrop_meta
from dnadesign.contracts.visual.yiu_payload_visual_v1 import (
    YiuPayloadMismatchV1,
    YiuPayloadMotifLayerV1,
)
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload
from dnadesign.cruncher.yiu.view_common import YIU_EMPTY_ROW_LABELS


def build_payload_mismatch_annotations(normalized: NormalizedPayload) -> list[YiuPayloadMismatchV1]:
    return [
        YiuPayloadMismatchV1(
            payload_index=entry.payload_index,
            junction_offset=entry.junction_offset,
            mutated_strand=entry.mutated_strand,
            native_base=entry.native_base,
            mutated_base=entry.mutated_base,
            opposing_base=entry.opposing_base,
        )
        for entry in normalized.mismatches
    ]


def build_payload_motif_layers(normalized: NormalizedPayload) -> list[YiuPayloadMotifLayerV1]:
    return [
        YiuPayloadMotifLayerV1(
            motif_instance_id=motif.motif_instance_id,
            tf_name=motif.tf_name,
            motif_name=motif.motif_name,
            reference_strand=motif.reference_strand,
            start=motif.start,
            end=motif.end,
            label=f"{motif.tf_name} ({motif.reference_strand})",
            matrix=[list(row) for row in motif.probabilities.rows],
        )
        for motif in normalized.motif_context.motifs
    ]


def build_payload_view_meta(normalized: NormalizedPayload) -> dict[str, object]:
    return {
        "payload_label": normalized.payload_label,
        "site_label": normalized.site_label,
        "row_labels": YIU_EMPTY_ROW_LABELS,
        "pwm_effective": normalized.motif_context.effective,
        "motif_ids": [motif.motif_instance_id for motif in normalized.motif_context.motifs],
        **build_sequence_evidence_span_backdrop_meta(
            start=normalized.junction.start,
            end=normalized.junction.end,
            coordinate_space="payload_forward",
        ),
    }


__all__ = [
    "build_payload_mismatch_annotations",
    "build_payload_motif_layers",
    "build_payload_view_meta",
]
