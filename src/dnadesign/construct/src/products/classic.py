"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/products/classic.py

Template-backed Construct product builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from dnadesign.usr import compute_id, normalize_sequence

from ..contracts.config import JobConfig, OutputVariantConfig
from ..contracts.errors import ValidationError
from ..persistence.records import BuiltRecord
from ..realization.assembly import assemble_full_construct, extract_output_sequence, resolve_anchor_part
from ..realization.parts import RealizedPart
from ..realization.placement import PlacementPlan
from ..realization.sequences import alphabet_for_sequence
from ..realization.slots import (
    assembly_mode,
    input_slot_sequence_length,
    oriented_slot_span_records,
    relative_slot_bounds,
    require_required_slot_bounds,
    slot_span_records,
)
from ..sequences.orientation import reverse_complement, reverse_complement_anchor_bounds
from ..sources.input_rows import input_fields, input_usr_labels
from .sequence_views import append_variant_label_suffix, build_variant_sequence_view


class ClassicTemplate(Protocol):
    id: str
    kind: str
    sequence: str
    source: str
    dataset: str | None
    field: str | None
    record_id: str | None
    circular: bool


def build_classic_record(
    *,
    row: dict[str, object],
    cfg: JobConfig,
    template: ClassicTemplate,
    template_sha256: str,
    spec_id: str,
    ordered_placements: list[PlacementPlan],
) -> BuiltRecord:
    if cfg.job.realize is None or cfg.job.template is None:
        raise ValidationError("job.template and job.realize are required when job.mode='realize_template'.")
    realize = cfg.job.realize
    created_at = datetime.now(timezone.utc).isoformat()
    assembled = assemble_full_construct(
        template,
        ordered_placements,
        row,
    )
    window = realize.window
    output_sequence, window_start, window_end = extract_output_sequence(
        full_construct=assembled.sequence,
        realized_parts=assembled.parts_by_name,
        cfg=cfg,
    )
    alphabet = alphabet_for_sequence(output_sequence)
    sequence_norm = normalize_sequence(output_sequence, "dna", alphabet)
    output_id = compute_id("dna", sequence_norm)
    label_primary, label_aliases = input_usr_labels(row)

    scanned_input_fields = [field for field in input_fields(cfg) if field != "id"]
    focal_part = assembled.parts_by_name.get(realize.focal_part or "")
    anchor_part = resolve_anchor_part(
        realized_parts=assembled.parts_by_name,
        ordered_realized_parts=assembled.ordered_parts,
        focal_part_name=realize.focal_part,
    )
    anchor_start, anchor_end = relative_slot_bounds(
        part=anchor_part,
        output_length=len(output_sequence),
        full_construct_length=len(assembled.sequence),
        window_start=window_start,
        mode=realize.mode,
    )
    require_window_anchor_handoff_bounds(
        row_id=row.get("id"),
        anchor_part=anchor_part,
        anchor_start=anchor_start,
        anchor_end=anchor_end,
        cfg=cfg,
    )
    slot_spans = slot_span_records(
        parts=assembled.ordered_parts,
        output_length=len(output_sequence),
        full_construct_length=len(assembled.sequence),
        window_start=window_start,
        mode=realize.mode,
    )
    require_required_slot_bounds(
        row_id=row.get("id"),
        required_slots=list(realize.required_slots),
        slot_spans=slot_spans,
        mode=realize.mode,
        window=realize.window,
    )
    metadata = {
        "id": output_id,
        "construct__job": cfg.job.id,
        "construct__spec_id": spec_id,
        "construct__context_id": f"{cfg.job.id}:{template.id}",
        "construct__context_kind": "template",
        "construct__template_id": template.id,
        "construct__template_kind": template.kind,
        "construct__template_source": template.source,
        "construct__template_dataset": template.dataset or "",
        "construct__template_field": template.field or "",
        "construct__template_record_id": template.record_id or "",
        "construct__template_sha256": template_sha256,
        "construct__template_length": len(template.sequence),
        "construct__template_circular": bool(template.circular),
        "construct__input_dataset": cfg.job.input.source.dataset,
        "construct__input_fields": scanned_input_fields,
        "construct__input_id": str(row["id"]),
        "construct__input_length": input_length_for_row(
            row=row,
            cfg=cfg,
            ordered_realized_parts=assembled.ordered_parts,
        ),
        "construct__assembly_mode": assembly_mode(assembled.ordered_parts),
        "construct__slot_count": len(assembled.ordered_parts),
        "construct__slots": slot_spans,
        "construct__anchor_id": str(row["id"]),
        "construct__anchor_orientation": anchor_part.orientation if anchor_part is not None else "",
        "construct__anchor_start": anchor_start,
        "construct__anchor_end": anchor_end,
        "construct__orientation": "forward",
        "construct__forward_anchor_start": anchor_start,
        "construct__forward_anchor_end": anchor_end,
        "construct__parent_forward_construct_id": "",
        "construct__mode": realize.mode,
        "construct__focal_part": realize.focal_part or "",
        "construct__focal_part_length": len(focal_part.sequence) if focal_part is not None else None,
        "construct__window_semantics": window.semantics if window is not None else "",
        "construct__window_reference": window.reference if window is not None else "",
        "construct__window_direction": window.direction if window is not None else "",
        "construct__window_size_bp": int(window.size_bp) if window is not None and window.size_bp is not None else None,
        "construct__window_upstream_bp": (
            int(window.upstream_bp) if window is not None and window.upstream_bp is not None else None
        ),
        "construct__window_downstream_bp": (
            int(window.downstream_bp) if window is not None and window.downstream_bp is not None else None
        ),
        "construct__window_offset_bp": (
            int(window.offset_bp) if window is not None and window.semantics == "fixed_total" else None
        ),
        "construct__window_start": window_start,
        "construct__window_end": window_end,
        "construct__resolved_length": len(output_sequence),
        "construct__full_construct_length": len(assembled.sequence),
        "construct__parts": [
            {
                "name": part.name,
                "role": part.role,
                "sequence_source": part.sequence_source,
                "sequence_field": part.sequence_field or "",
                "placement_kind": part.kind,
                "orientation": part.orientation,
                "template_start": part.start,
                "template_end": part.end,
                "realized_start": part.realized_start,
                "realized_end": part.realized_end,
                "length": len(part.sequence),
            }
            for part in assembled.ordered_parts
        ],
    }
    return BuiltRecord(
        output_id=output_id,
        sequence=output_sequence,
        alphabet=alphabet,
        metadata=metadata,
        label_primary=label_primary,
        label_aliases=label_aliases,
        created_at=created_at,
    )


def build_variant_record(
    *,
    forward_record: BuiltRecord,
    variant: OutputVariantConfig,
    output_dataset_id: str,
) -> BuiltRecord:
    forward_slots = list(forward_record.metadata.get("construct__slots") or [])
    if variant.orientation == "forward":
        sequence = forward_record.sequence
        anchor_start = optional_int(forward_record.metadata.get("construct__forward_anchor_start"))
        anchor_end = optional_int(forward_record.metadata.get("construct__forward_anchor_end"))
        parent_forward_construct_id = ""
        oriented_slots = forward_slots
    else:
        sequence = reverse_complement(forward_record.sequence)
        anchor_start, anchor_end = reverse_complement_optional_anchor_bounds(
            sequence_length=len(forward_record.sequence),
            anchor_start_0=optional_int(forward_record.metadata.get("construct__forward_anchor_start")),
            anchor_end_0=optional_int(forward_record.metadata.get("construct__forward_anchor_end")),
        )
        parent_forward_construct_id = forward_record.output_id
        oriented_slots = oriented_slot_span_records(
            forward_slots=forward_slots,
            output_length=len(forward_record.sequence),
            orientation=variant.orientation,
        )
    alphabet = alphabet_for_sequence(sequence)
    output_id = compute_id("dna", normalize_sequence(sequence, "dna", alphabet))
    metadata = dict(forward_record.metadata)
    metadata.update(
        {
            "id": output_id,
            "construct__anchor_start": anchor_start,
            "construct__anchor_end": anchor_end,
            "construct__orientation": variant.orientation,
            "construct__parent_forward_construct_id": parent_forward_construct_id,
            "construct__slots": oriented_slots,
        }
    )
    label_suffix = (
        "realized_context_forward" if variant.orientation == "forward" else "realized_context_reverse_complement"
    )
    label_primary = append_variant_label_suffix(forward_record.label_primary, label_suffix)
    label_aliases = [
        alias
        for alias in (append_variant_label_suffix(alias, label_suffix) for alias in forward_record.label_aliases)
        if alias is not None
    ]
    record = BuiltRecord(
        output_id=output_id,
        sequence=sequence,
        alphabet=alphabet,
        metadata=metadata,
        label_primary=label_primary,
        label_aliases=label_aliases,
        created_at=forward_record.created_at,
    )
    view_anchor_start = anchor_start
    view_anchor_end = anchor_end
    view_forward_anchor_start = optional_int(forward_record.metadata.get("construct__forward_anchor_start"))
    view_forward_anchor_end = optional_int(forward_record.metadata.get("construct__forward_anchor_end"))
    if variant.anchor_part is not None:
        view_anchor_start, view_anchor_end, view_forward_anchor_start, view_forward_anchor_end = slot_anchor_bounds(
            slots=oriented_slots,
            slot_id=variant.anchor_part,
            output_id=output_id,
        )
        if variant.anchor_window_size_bp is not None:
            view_anchor_start, view_anchor_end = fixed_anchor_window_bounds(
                anchor_start_0=view_anchor_start,
                anchor_end_0=view_anchor_end,
                sequence_length=len(sequence),
                window_size_bp=variant.anchor_window_size_bp,
                slot_id=variant.anchor_part,
                output_id=output_id,
            )
            view_forward_anchor_start, view_forward_anchor_end = fixed_anchor_window_bounds(
                anchor_start_0=view_forward_anchor_start,
                anchor_end_0=view_forward_anchor_end,
                sequence_length=len(forward_record.sequence),
                window_size_bp=variant.anchor_window_size_bp,
                slot_id=variant.anchor_part,
                output_id=output_id,
            )
    view_aliases = list(record.label_aliases)
    if variant.view_name is not None:
        view_aliases = [
            alias
            for alias in (append_variant_label_suffix(alias, variant.view_name) for alias in record.label_aliases)
            if alias is not None
        ]
    record.sequence_view = build_variant_sequence_view(
        record=record,
        output_dataset_id=output_dataset_id,
        context_kind=variant.context_kind,
        recommended_pooling=variant.recommended_pooling,
        anchor_start_0=view_anchor_start,
        anchor_end_0=view_anchor_end,
        forward_anchor_start_0=view_forward_anchor_start,
        forward_anchor_end_0=view_forward_anchor_end,
        view_name=variant.view_name,
        aliases=view_aliases,
    )
    return record


def input_length_for_row(
    *,
    row: dict[str, object],
    cfg: JobConfig,
    ordered_realized_parts: list[RealizedPart],
) -> int:
    if cfg.job.input.field is not None:
        raw = row.get(cfg.job.input.field)
        if raw is None:
            raise ValidationError(f"Input row '{row.get('id')}' is missing field '{cfg.job.input.field}'.")
        return len(str(raw).strip())
    return input_slot_sequence_length(ordered_realized_parts)


def optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def reverse_complement_optional_anchor_bounds(
    *,
    sequence_length: int,
    anchor_start_0: int | None,
    anchor_end_0: int | None,
) -> tuple[int | None, int | None]:
    if anchor_start_0 is None and anchor_end_0 is None:
        return None, None
    if anchor_start_0 is None or anchor_end_0 is None:
        raise ValidationError("Construct anchor bounds must provide both start and end when present.")
    return reverse_complement_anchor_bounds(
        sequence_length=sequence_length,
        anchor_start_0=anchor_start_0,
        anchor_end_0=anchor_end_0,
    )


def slot_anchor_bounds(
    *,
    slots: list[dict[str, object]],
    slot_id: str,
    output_id: str,
) -> tuple[int, int, int, int]:
    for slot in slots:
        if str(slot.get("slot_id") or "") != slot_id:
            continue
        start = optional_int(slot.get("start"))
        end = optional_int(slot.get("end"))
        forward_start = optional_int(slot.get("forward_start"))
        forward_end = optional_int(slot.get("forward_end"))
        if None in {start, end, forward_start, forward_end}:
            raise ValidationError(
                f"output_variants anchor_part '{slot_id}' does not have contiguous emitted bounds for {output_id}."
            )
        return int(start), int(end), int(forward_start), int(forward_end)
    raise ValidationError(
        f"output_variants anchor_part '{slot_id}' is not present in construct__slots for {output_id}."
    )


def fixed_anchor_window_bounds(
    *,
    anchor_start_0: int,
    anchor_end_0: int,
    sequence_length: int,
    window_size_bp: int,
    slot_id: str,
    output_id: str,
) -> tuple[int, int]:
    anchor_length = anchor_end_0 - anchor_start_0
    if anchor_length > window_size_bp:
        raise ValidationError(
            f"output_variants anchor_part '{slot_id}' span length {anchor_length} exceeds "
            f"anchor_window_size_bp={window_size_bp} for {output_id}."
        )
    if window_size_bp > sequence_length:
        raise ValidationError(
            f"output_variants anchor_window_size_bp={window_size_bp} exceeds sequence length "
            f"{sequence_length} for {output_id}."
        )
    ideal_start = anchor_start_0 + ((anchor_length - window_size_bp) // 2)
    window_start = min(max(ideal_start, 0), sequence_length - window_size_bp)
    window_end = window_start + window_size_bp
    if window_start > anchor_start_0 or window_end < anchor_end_0:
        raise ValidationError(
            f"output_variants anchor_window_size_bp={window_size_bp} cannot contain "
            f"anchor_part '{slot_id}' span [{anchor_start_0},{anchor_end_0}) for {output_id}."
        )
    return window_start, window_end


def require_window_anchor_handoff_bounds(
    *,
    row_id: object,
    anchor_part: RealizedPart | None,
    anchor_start: int | None,
    anchor_end: int | None,
    cfg: JobConfig,
) -> None:
    if cfg.job.realize is None:
        raise ValidationError("job.realize is required when checking construct anchor handoff bounds.")
    realize = cfg.job.realize
    if anchor_part is None or realize.mode == "full_construct":
        return
    if anchor_start is not None and anchor_end is not None:
        return
    window = realize.window
    window_desc = "window"
    if window is not None:
        if window.semantics == "fixed_total":
            window_desc = (
                f"fixed_total(reference={window.reference}, direction={window.direction}, size_bp={window.size_bp})"
            )
        else:
            window_desc = f"anchor_plus_context(upstream_bp={window.upstream_bp}, downstream_bp={window.downstream_bp})"
    raise ValidationError(
        "Construct window does not preserve the focal anchor as one contiguous span in the emitted sequence, "
        "so construct__anchor_start/end cannot be emitted for downstream infer handoff. "
        f"row_id={row_id} anchor={anchor_part.name} mode={realize.mode} window={window_desc}. "
        "Choose full_construct, anchor_plus_context, or a fixed_total window that contains the full anchor span."
    )
