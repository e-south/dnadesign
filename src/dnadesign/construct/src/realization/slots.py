"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/realization/slots.py

Named-slot lineage and emitted-coordinate contracts for construct realization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Protocol

from ..contracts.errors import ValidationError
from ..sequences.orientation import reverse_complement_anchor_bounds


class RealizedSlotPart(Protocol):
    name: str
    role: str
    kind: str
    sequence_source: str
    sequence_field: str | None
    orientation: str
    start: int
    end: int
    sequence: str
    realized_start: int
    realized_end: int


class WindowLike(Protocol):
    semantics: str
    reference: str
    direction: str
    size_bp: int | None
    upstream_bp: int | None
    downstream_bp: int | None


def relative_slot_bounds(
    *,
    part: RealizedSlotPart | None,
    output_length: int,
    full_construct_length: int,
    window_start: int,
    mode: str,
) -> tuple[int | None, int | None]:
    if part is None:
        return None, None
    if mode == "full_construct":
        return part.realized_start, part.realized_end

    start = (part.realized_start - window_start) % full_construct_length
    end = start + len(part.sequence)
    if end > output_length:
        return None, None
    return start, end


def input_slot_parts(parts: list[RealizedSlotPart]) -> list[RealizedSlotPart]:
    return [part for part in parts if part.sequence_source == "input_field"]


def assembly_mode(parts: list[RealizedSlotPart]) -> str:
    return "multi_slot" if len(input_slot_parts(parts)) > 1 else "single_slot"


def input_slot_sequence_length(parts: list[RealizedSlotPart]) -> int:
    return sum(len(part.sequence) for part in input_slot_parts(parts))


def slot_span_records(
    *,
    parts: list[RealizedSlotPart],
    output_length: int,
    full_construct_length: int,
    window_start: int,
    mode: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for part in parts:
        start, end = relative_slot_bounds(
            part=part,
            output_length=output_length,
            full_construct_length=full_construct_length,
            window_start=window_start,
            mode=mode,
        )
        records.append(
            {
                "slot_id": part.name,
                "role": part.role,
                "sequence_source": part.sequence_source,
                "sequence_field": part.sequence_field or "",
                "placement_kind": part.kind,
                "orientation": part.orientation,
                "template_start": part.start,
                "template_end": part.end,
                "forward_start": start,
                "forward_end": end,
                "start": start,
                "end": end,
                "length": len(part.sequence),
            }
        )
    return records


def require_required_slot_bounds(
    *,
    row_id: object,
    required_slots: list[str],
    slot_spans: list[dict[str, object]],
    mode: str,
    window: WindowLike | None,
) -> None:
    if not required_slots:
        return
    by_id = {str(slot["slot_id"]): slot for slot in slot_spans}
    for slot_id in required_slots:
        slot = by_id.get(slot_id)
        if slot is None:
            raise ValidationError(f"Construct required slot '{slot_id}' is not defined for row_id={row_id}.")
        if slot.get("start") is not None and slot.get("end") is not None:
            continue
        raise ValidationError(
            f"Construct window does not preserve required slot '{slot_id}' as one contiguous span in the emitted "
            f"sequence. row_id={row_id} mode={mode} window={_window_description(mode=mode, window=window)}. "
            "Choose full_construct, a larger window, or remove the slot from realize.required_slots explicitly."
        )


def oriented_slot_span_records(
    *,
    forward_slots: list[dict[str, object]],
    output_length: int,
    orientation: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for slot in forward_slots:
        record = dict(slot)
        forward_start = slot.get("forward_start")
        forward_end = slot.get("forward_end")
        if orientation == "reverse_complement" and forward_start is not None and forward_end is not None:
            start, end = reverse_complement_anchor_bounds(
                sequence_length=output_length,
                anchor_start_0=int(forward_start),
                anchor_end_0=int(forward_end),
            )
            record["start"] = start
            record["end"] = end
        records.append(record)
    return records


def _window_description(*, mode: str, window: WindowLike | None) -> str:
    if window is None:
        return mode
    if window.semantics == "fixed_total":
        return f"fixed_total(reference={window.reference}, direction={window.direction}, size_bp={window.size_bp})"
    return f"anchor_plus_context(upstream_bp={window.upstream_bp}, downstream_bp={window.downstream_bp})"
