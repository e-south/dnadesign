"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/view_sequence_metadata.py

Metadata policy for split and assembled payload sequence-view contracts.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.yiu.bsmbi import GhostExcisedContext, SplitFragmentDisplaySpec
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload
from dnadesign.cruncher.yiu.view_common import YIU_EMPTY_ROW_LABELS


def _build_connector_meta(
    *,
    span: dict[str, object],
    cross_indices: list[int],
) -> dict[str, object]:
    start = int(span["start"])
    end = int(span["end"])
    crossed = set(cross_indices)
    return {
        "connector_hidden_indices": [index for index in range(start, end) if index not in crossed],
        "connector_cross_indices": cross_indices,
        "connector_overhang_spans": [span],
    }


def _ghost_dim_base_indices(ghost_context: GhostExcisedContext | None) -> dict[str, list[int]]:
    if ghost_context is None:
        return {"primary": [], "complement": []}
    return {
        "primary": list(ghost_context.primary_indices),
        "complement": list(ghost_context.complement_indices),
    }


def build_split_payload_row_meta(fragment: SplitFragmentDisplaySpec) -> dict[str, object]:
    sticky_end_span = fragment.sticky_end_display_span.model_dump(mode="json")
    ghost_context = fragment.ghost_excised_context
    return {
        "view_id": "split_payload",
        "fragment_side": fragment.fragment_side,
        "panel_order": fragment.panel_order,
        "retained_primary_sequence_5to3": fragment.retained_primary_sequence_5to3,
        "retained_complement_sequence_3to5": fragment.retained_complement_sequence_3to5,
        "retained_payload_body_sequence_5to3": fragment.retained_payload_body_sequence_5to3,
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
        "dim_base_indices": _ghost_dim_base_indices(ghost_context),
        **_build_connector_meta(span=sticky_end_span, cross_indices=[]),
    }


def build_assembled_payload_view_meta(normalized: NormalizedPayload) -> dict[str, object]:
    highlight_indices = [site.payload_index for site in normalized.mismatches]
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
        "base_highlights": {"primary": highlight_indices, "complement": highlight_indices},
        "row_labels": YIU_EMPTY_ROW_LABELS,
        **_build_connector_meta(span=junction_span, cross_indices=highlight_indices),
    }


__all__ = [
    "YIU_EMPTY_ROW_LABELS",
    "build_assembled_payload_view_meta",
    "build_split_payload_row_meta",
]
