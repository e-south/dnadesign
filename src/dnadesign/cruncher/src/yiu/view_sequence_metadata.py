"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/view_sequence_metadata.py

Metadata policy for split and assembled payload sequence-view contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.contracts.visual.sequence_evidence_meta import build_sequence_evidence_connector_span_meta
from dnadesign.cruncher.yiu.bsmbi import GhostExcisedContext, SplitFragmentDisplaySpec
from dnadesign.cruncher.yiu.domain_models import MismatchSelection, NormalizedPayload
from dnadesign.cruncher.yiu.view_common import YIU_EMPTY_ROW_LABELS

YIU_MISMATCH_HIGHLIGHT_COLOR = "#B91C1C"


def _ghost_dim_base_indices(ghost_context: GhostExcisedContext | None) -> dict[str, list[int]]:
    if ghost_context is None:
        return {"primary": [], "complement": []}
    return {
        "primary": list(ghost_context.primary_indices),
        "complement": list(ghost_context.complement_indices),
    }


def _mismatch_highlight_indices(mismatches: list[MismatchSelection]) -> dict[str, list[int]]:
    highlights = {"primary": [], "complement": []}
    for mismatch in mismatches:
        row_id = "primary" if mismatch.mutated_strand == "payload" else "complement"
        highlights[row_id].append(mismatch.payload_index)
    for row_id in highlights:
        highlights[row_id].sort()
    return highlights


def _split_mismatch_highlight_indices(
    fragment: SplitFragmentDisplaySpec,
    normalized: NormalizedPayload,
) -> dict[str, list[int]]:
    highlights = {"primary": [], "complement": []}
    sticky_start = fragment.sticky_end_display_span.start
    junction_end = normalized.junction.end
    for mismatch in normalized.mismatches:
        display_index = sticky_start + (junction_end - 1 - mismatch.payload_index)
        row_id = "primary" if mismatch.mutated_strand == "complement" else "complement"
        highlights[row_id].append(display_index)
    for row_id in highlights:
        highlights[row_id].sort()
    return highlights


def build_split_payload_row_meta(
    fragment: SplitFragmentDisplaySpec,
    normalized: NormalizedPayload,
) -> dict[str, object]:
    sticky_end_span = fragment.sticky_end_display_span.model_dump(mode="json")
    ghost_context = fragment.ghost_excised_context
    return {
        "view_id": "split_payload",
        "fragment_side": fragment.fragment_side,
        "panel_order": fragment.panel_order,
        "payload_body_sequence_5to3": fragment.payload_body_sequence_5to3,
        "display_payload_body_sequence_5to3": fragment.display_payload_body_sequence_5to3,
        "retained_primary_sequence_5to3": fragment.retained_primary_sequence_5to3,
        "retained_complement_sequence_3to5": fragment.retained_complement_sequence_3to5,
        "selected_sticky_end_sequence_5to3": fragment.selected_sticky_end_sequence_5to3,
        "canonical_sticky_end_sequence_5to3": fragment.canonical_sticky_end_sequence_5to3,
        "sticky_end_display_span": sticky_end_span,
        "payload_body_display_span": fragment.payload_body_display_span.model_dump(mode="json"),
        "retained_primary_display_span": fragment.retained_primary_display_span.model_dump(mode="json"),
        "retained_complement_display_span": fragment.retained_complement_display_span.model_dump(mode="json"),
        "payload_junction_window": fragment.payload_junction_window.model_dump(mode="json"),
        "sticky_end_orientation": fragment.sticky_end_orientation,
        "recognition_site_orientation": fragment.recognition_site_orientation,
        "ghost_excised_context": None if ghost_context is None else ghost_context.model_dump(mode="json"),
        "row_labels": YIU_EMPTY_ROW_LABELS,
        "base_highlights": _split_mismatch_highlight_indices(fragment, normalized),
        "base_highlight_color": YIU_MISMATCH_HIGHLIGHT_COLOR,
        "dim_base_indices": _ghost_dim_base_indices(ghost_context),
        **build_sequence_evidence_connector_span_meta(
            start=sticky_end_span["start"],
            end=sticky_end_span["end"],
            coordinate_space=sticky_end_span.get("coordinate_space"),
        ),
    }


def build_assembled_payload_view_meta(normalized: NormalizedPayload) -> dict[str, object]:
    junction_span = {
        "start": normalized.junction.start,
        "end": normalized.junction.end,
        "coordinate_space": "payload_forward",
    }
    return {
        "view_id": "assembled_payload",
        "junction_span": junction_span,
        "mismatches": [site.model_dump(mode="json") for site in normalized.mismatches],
        "sequence_identity_to_reference_payload": normalized.selected_payload_sequence
        == normalized.reference_payload_sequence,
        "base_highlights": _mismatch_highlight_indices(normalized.mismatches),
        "base_highlight_color": YIU_MISMATCH_HIGHLIGHT_COLOR,
        "row_labels": YIU_EMPTY_ROW_LABELS,
        **build_sequence_evidence_connector_span_meta(
            start=junction_span["start"],
            end=junction_span["end"],
            coordinate_space="payload_forward",
        ),
    }


__all__ = [
    "YIU_EMPTY_ROW_LABELS",
    "build_assembled_payload_view_meta",
    "build_split_payload_row_meta",
]
