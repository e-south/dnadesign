"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/view_sequence_contracts.py

Sequence-evidence contract shells for split and assembled YIU views.

Module Author(s): OpenAI Codex
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


def _build_sequence_contract(
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


def build_split_payload_view_rows(normalized: NormalizedPayload) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fragment in sorted(build_split_fragment_display_specs(normalized), key=lambda item: item.panel_order):
        rows.append(
            _build_sequence_contract(
                state_id=f"split_payload_{fragment.fragment_side}",
                title=fragment.title,
                sequence=fragment.display_primary_sequence_5to3,
                complement_sequence=fragment.display_complement_sequence_3to5,
                meta=build_split_payload_row_meta(fragment),
            )
        )
    return rows


def build_assembled_payload_view_contract(normalized: NormalizedPayload) -> dict[str, object]:
    return _build_sequence_contract(
        state_id="assembled_payload",
        title="Assembled payload",
        sequence=normalized.selected_payload_sequence,
        complement_sequence=assembled_payload_aligned_complement_3to5(normalized),
        meta=build_assembled_payload_view_meta(normalized),
    )


__all__ = [
    "YIU_EMPTY_ROW_LABELS",
    "build_assembled_payload_view_contract",
    "build_split_payload_view_rows",
]
