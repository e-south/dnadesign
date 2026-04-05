"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/view_registry.py

Canonical view registry for payload-centric YIU bundle publication and render
planning.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.cruncher.yiu.bundle_models import PayloadViewEntry


@dataclass(frozen=True)
class PayloadViewDefinition:
    view_id: str
    visual_direction: str
    contract_kind: str
    adapter_kind: str
    input_kind: str
    renderer_kind: str


_CANONICAL_PAYLOAD_VIEW_DEFINITIONS: tuple[PayloadViewDefinition, ...] = (
    PayloadViewDefinition(
        view_id="payload",
        visual_direction="evidence_ribbon",
        contract_kind="yiu_payload_visual_v1",
        adapter_kind="yiu_payload_visual_v1",
        input_kind="json",
        renderer_kind="nucleotide_evidence_map",
    ),
    PayloadViewDefinition(
        view_id="split_payload",
        visual_direction="operator_strip",
        contract_kind="sequence_evidence_map_v1",
        adapter_kind="sequence_evidence_map_v1",
        input_kind="jsonl",
        renderer_kind="sequence_rows",
    ),
    PayloadViewDefinition(
        view_id="assembled_payload",
        visual_direction="operator_strip",
        contract_kind="sequence_evidence_map_v1",
        adapter_kind="sequence_evidence_map_v1",
        input_kind="json",
        renderer_kind="nucleotide_evidence_map",
    ),
)

_PAYLOAD_VIEW_DEFINITIONS_BY_ID = {definition.view_id: definition for definition in _CANONICAL_PAYLOAD_VIEW_DEFINITIONS}


def canonical_payload_view_definitions() -> tuple[PayloadViewDefinition, ...]:
    return _CANONICAL_PAYLOAD_VIEW_DEFINITIONS


def payload_view_definition(view_id: str) -> PayloadViewDefinition:
    try:
        return _PAYLOAD_VIEW_DEFINITIONS_BY_ID[view_id]
    except KeyError as exc:
        supported = ", ".join(sorted(_PAYLOAD_VIEW_DEFINITIONS_BY_ID))
        raise ValueError(f"unsupported YIU view id {view_id!r}; expected one of: {supported}") from exc


def validate_payload_view_entry(entry: PayloadViewEntry) -> PayloadViewDefinition:
    definition = payload_view_definition(entry.view_id)
    mismatches: list[str] = []
    if entry.visual_direction != definition.visual_direction:
        mismatches.append(f"visual_direction={entry.visual_direction!r} expected {definition.visual_direction!r}")
    if entry.contract_kind != definition.contract_kind:
        mismatches.append(f"contract_kind={entry.contract_kind!r} expected {definition.contract_kind!r}")
    if entry.input_kind != definition.input_kind:
        mismatches.append(f"input_kind={entry.input_kind!r} expected {definition.input_kind!r}")
    if entry.renderer_kind != definition.renderer_kind:
        mismatches.append(f"renderer_kind={entry.renderer_kind!r} expected {definition.renderer_kind!r}")
    if mismatches:
        detail = ", ".join(mismatches)
        raise ValueError(f"YIU view entry drift for {entry.view_id!r}: {detail}")
    return definition


__all__ = [
    "canonical_payload_view_definitions",
    "payload_view_definition",
    "PayloadViewDefinition",
    "validate_payload_view_entry",
]
