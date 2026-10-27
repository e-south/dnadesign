"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/yiu/view_sequence_contracts.py

Sequence-evidence contract shells for split and assembled YIU views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.contracts.visual import SequenceEvidenceMapV1
from dnadesign.cruncher.yiu.bsmbi import (
    assembled_payload_aligned_complement_3to5,
    build_split_fragment_display_specs,
)
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload
from dnadesign.cruncher.yiu.view_common import YIU_EMPTY_ROW_LABELS
from dnadesign.cruncher.yiu.view_sequence_metadata import (
    build_assembled_payload_view_meta,
    build_split_payload_row_meta,
)
from dnadesign.cruncher.yiu.view_styles import build_assembled_payload_view_title


def _build_sequence_contract(
    *,
    state_id: str,
    title: str,
    sequence: str,
    complement_sequence: str,
    boundaries: list[dict[str, object]],
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
            "boundaries": boundaries,
            "pairings": [],
            "display": {"title": title},
            "meta": meta,
        }
    ).model_dump(mode="json")


def _build_ligation_junction_boundaries(*, start: int, end: int) -> list[dict[str, object]]:
    return [
        {
            "boundary_id": "junction_start",
            "row_id": "primary",
            "boundary": start,
            "boundary_kind": "ligation_junction",
            "display_label": "Junction start",
            "short_label": "",
        },
        {
            "boundary_id": "junction_end",
            "row_id": "complement",
            "boundary": end,
            "boundary_kind": "ligation_junction",
            "display_label": "Junction end",
            "short_label": "",
        },
    ]


def build_split_payload_view_rows(normalized: NormalizedPayload) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fragment in sorted(build_split_fragment_display_specs(normalized), key=lambda item: item.panel_order):
        rows.append(
            _build_sequence_contract(
                state_id=f"split_payload_{fragment.fragment_side}",
                title=fragment.title,
                sequence=fragment.display_primary_sequence_5to3,
                complement_sequence=fragment.display_complement_sequence_3to5,
                boundaries=_build_ligation_junction_boundaries(
                    start=fragment.sticky_end_display_span.start,
                    end=fragment.sticky_end_display_span.end,
                ),
                meta=build_split_payload_row_meta(fragment, normalized),
            )
        )
    return rows


def build_assembled_payload_view_contract(normalized: NormalizedPayload) -> dict[str, object]:
    return _build_sequence_contract(
        state_id="assembled_payload",
        title=build_assembled_payload_view_title(),
        sequence=normalized.selected_payload_sequence,
        complement_sequence=assembled_payload_aligned_complement_3to5(normalized),
        boundaries=_build_ligation_junction_boundaries(
            start=normalized.junction.start,
            end=normalized.junction.end,
        ),
        meta=build_assembled_payload_view_meta(normalized),
    )


__all__ = [
    "YIU_EMPTY_ROW_LABELS",
    "build_assembled_payload_view_contract",
    "build_split_payload_view_rows",
]
