"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/view_contracts.py

Pure YIU v4 view-contract and style builders shared by publish and integrity.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

from dnadesign.baserender import cruncher_showcase_style_overrides
from dnadesign.contracts.visual import SequenceEvidenceMapV1, YiuPayloadVisualV1
from dnadesign.contracts.visual.yiu_payload_visual_v1 import (
    YiuPayloadDisplayV1,
    YiuPayloadJunctionV1,
    YiuPayloadMismatchV1,
    YiuPayloadMotifLayerV1,
)
from dnadesign.cruncher.yiu.bsmbi import (
    assembled_payload_aligned_complement_3to5,
    build_split_fragment_display_specs,
)
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload

YIU_EMPTY_ROW_LABELS: dict[str, str] = {}
_YIU_FIGURE_SCALE = 1.24


def _pretty_label(text: str | None) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    normalized = re.sub(r"[_-]+", " ", raw)
    return " ".join(token[:1].upper() + token[1:] for token in normalized.split())


def build_payload_view_title(normalized: NormalizedPayload) -> str:
    motif_tfs = sorted({motif.tf_name for motif in normalized.motif_context.motifs if str(motif.tf_name).strip()})
    if len(motif_tfs) == 1:
        tf_label = _pretty_label(motif_tfs[0])
        motif_count = len(normalized.motif_context.motifs)
        suffix = f" ({motif_count} sites)" if motif_count > 1 else ""
        return f"{tf_label} payload{suffix}"
    if normalized.payload_label:
        return _pretty_label(normalized.payload_label)
    return _pretty_label(normalized.name) or "Payload"


def build_yiu_style_overrides(view_id: str) -> dict[str, object]:
    if view_id == "payload":
        base = dict(cruncher_showcase_style_overrides())
        base["figure_scale"] = _YIU_FIGURE_SCALE
        base["padding_x"] = 42.0
        base["padding_y"] = 24.0
        base["font_size_seq"] = 13
        base["font_size_label"] = 11
        base["legend"] = False
        base["connectors"] = True
        base["connector_width"] = 1.1
        base["connector_alpha"] = 0.78
        base["connector_dash"] = ()
        return base

    base: dict[str, object] = {
        "figure_scale": _YIU_FIGURE_SCALE,
        "padding_x": 42.0,
        "padding_y": 24.0,
        "font_size_seq": 13,
        "font_size_label": 11,
        "legend_font_size": 10,
        "legend_gap_x": 10.0,
        "legend_height_px": 52.0,
        "layout": {"outer_pad_cells": 0.18},
        "sequence": {"strand_gap_cells": 0.22, "to_kmer_gap_cells": 0.18},
        "kmer": {"box_height_cells": 1.02, "fill_alpha": 0.94, "text_y_nudge_cells": 0.0},
        "overlay_align": "center",
        "connector_width": 1.1,
        "connector_alpha": 0.78,
        "connector_dash": (),
    }
    if view_id in {"payload", "split_payload", "assembled_payload"}:
        base["legend"] = False
    if view_id == "assembled_payload":
        base["padding_y"] = 28.0
    return base


def _span(*, start: int, end: int, coordinate_space: str) -> dict[str, object]:
    return {"start": start, "end": end, "coordinate_space": coordinate_space}


def _sequence_contract(
    *,
    state_id: str,
    title: str,
    sequence: str,
    complement_sequence: str,
    meta: dict[str, object],
) -> dict[str, object]:
    return SequenceEvidenceMapV1.model_validate(
        {
            "contract_kind": "sequence_evidence_map_v1",
            "state_id": state_id,
            "topology_kind": "linear_dsdna",
            "alphabet": "iupac_dna",
            "primary_sequence": sequence,
            "complement_sequence": complement_sequence,
            "owners": [],
            "effect_tags": [],
            "boundaries": [],
            "pairings": [],
            "display": {"title": title},
            "meta": meta,
        }
    ).model_dump(mode="json")


def build_payload_view_contract(normalized: NormalizedPayload) -> dict[str, object]:
    motif_layers = [
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
        mismatches=[
            YiuPayloadMismatchV1(
                payload_index=entry.payload_index,
                junction_offset=entry.junction_offset,
                mutated_strand=entry.mutated_strand,
                native_base=entry.native_base,
                mutated_base=entry.mutated_base,
                opposing_base=entry.opposing_base,
            )
            for entry in normalized.mismatches
        ],
        motif_layers=motif_layers,
        display=YiuPayloadDisplayV1(title=build_payload_view_title(normalized)),
        meta={
            "payload_label": normalized.payload_label,
            "site_label": normalized.site_label,
            "row_labels": YIU_EMPTY_ROW_LABELS,
            "pwm_effective": normalized.motif_context.effective,
            "motif_ids": [motif.motif_instance_id for motif in normalized.motif_context.motifs],
        },
    ).model_dump(mode="json")


def build_split_payload_view_rows(normalized: NormalizedPayload) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fragment in sorted(build_split_fragment_display_specs(normalized), key=lambda item: item.panel_order):
        span = fragment.sticky_end_display_span.model_dump(mode="json")
        ghost = fragment.ghost_excised_context.model_dump(mode="json") if fragment.ghost_excised_context else None
        dim_base_indices = (
            {
                "primary": list(fragment.ghost_excised_context.primary_indices),
                "complement": list(fragment.ghost_excised_context.complement_indices),
            }
            if fragment.ghost_excised_context is not None
            else {"primary": [], "complement": []}
        )
        rows.append(
            _sequence_contract(
                state_id=f"split_payload_{fragment.fragment_side}",
                title=fragment.title,
                sequence=fragment.display_primary_sequence_5to3,
                complement_sequence=fragment.display_complement_sequence_3to5,
                meta={
                    "view_id": "split_payload",
                    "fragment_side": fragment.fragment_side,
                    "panel_order": fragment.panel_order,
                    "retained_primary_sequence_5to3": fragment.retained_primary_sequence_5to3,
                    "retained_complement_sequence_3to5": fragment.retained_complement_sequence_3to5,
                    "retained_payload_body_sequence_5to3": fragment.retained_payload_body_sequence_5to3,
                    "selected_sticky_end_sequence_5to3": fragment.selected_sticky_end_sequence_5to3,
                    "canonical_sticky_end_sequence_5to3": fragment.canonical_sticky_end_sequence_5to3,
                    "sticky_end_display_span": span,
                    "payload_body_display_span": fragment.payload_body_display_span.model_dump(mode="json"),
                    "retained_primary_display_span": fragment.retained_primary_display_span.model_dump(mode="json"),
                    "retained_complement_display_span": fragment.retained_complement_display_span.model_dump(
                        mode="json"
                    ),
                    "payload_junction_window": fragment.payload_junction_window.model_dump(mode="json"),
                    "sticky_end_orientation": fragment.sticky_end_orientation,
                    "recognition_site_orientation": fragment.recognition_site_orientation,
                    "ghost_excised_context": ghost,
                    "row_labels": YIU_EMPTY_ROW_LABELS,
                    "dim_base_indices": dim_base_indices,
                    "connector_hidden_indices": list(range(span["start"], span["end"])),
                    "connector_cross_indices": [],
                    "connector_overhang_spans": [span],
                },
            )
        )
    return rows


def build_assembled_payload_view_contract(normalized: NormalizedPayload) -> dict[str, object]:
    highlight_indices = [site.payload_index for site in normalized.mismatches]
    hidden_indices = [
        index
        for index in range(normalized.junction.start, normalized.junction.end)
        if index not in set(highlight_indices)
    ]
    junction_span = _span(
        start=normalized.junction.start,
        end=normalized.junction.end,
        coordinate_space="payload_forward",
    )
    return _sequence_contract(
        state_id="assembled_payload",
        title="Assembled payload",
        sequence=normalized.selected_payload_sequence,
        complement_sequence=assembled_payload_aligned_complement_3to5(normalized),
        meta={
            "view_id": "assembled_payload",
            "junction_span": junction_span,
            "mismatches": [site.model_dump(mode="json") for site in normalized.mismatches],
            "sequence_identity_to_reference_payload": normalized.selected_payload_sequence
            == normalized.reference_payload_sequence,
            "base_highlights": {"primary": highlight_indices, "complement": highlight_indices},
            "connector_hidden_indices": hidden_indices,
            "connector_cross_indices": highlight_indices,
            "connector_overhang_spans": [junction_span],
            "row_labels": YIU_EMPTY_ROW_LABELS,
        },
    )


__all__ = [
    "YIU_EMPTY_ROW_LABELS",
    "build_assembled_payload_view_contract",
    "build_payload_view_contract",
    "build_payload_view_title",
    "build_split_payload_view_rows",
    "build_yiu_style_overrides",
]
