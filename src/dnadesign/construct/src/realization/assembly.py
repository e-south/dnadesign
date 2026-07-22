"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/realization/assembly.py

Sequence assembly and emitted-window extraction for Construct realization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..contracts.config import JobConfig, PartConfig
from ..contracts.errors import ValidationError
from ..sequences.orientation import reverse_complement
from .parts import RealizedPart
from .placement import PlacementPlan, validate_placement_guards
from .sequences import ensure_dna_text
from .windows import normalize_window_geometry


class AssemblyTemplate(Protocol):
    sequence: str
    circular: bool


@dataclass(frozen=True)
class AssembledConstruct:
    sequence: str
    ordered_parts: list[RealizedPart]
    parts_by_name: dict[str, RealizedPart]


def assemble_full_construct(
    template: AssemblyTemplate,
    placements: list[PlacementPlan],
    row: dict[str, object],
) -> AssembledConstruct:
    template_seq = template.sequence
    cursor = 0
    out: list[str] = []
    out_len = 0
    realized: dict[str, RealizedPart] = {}
    realized_ordered: list[RealizedPart] = []

    for resolved in placements:
        part = resolved.part
        site = resolved.site
        validate_placement_guards(
            template=template,
            part=part,
            site=site,
        )
        prefix = template_seq[cursor : site.start]
        out.append(prefix)
        out_len += len(prefix)

        seq = part_sequence(part, row)
        realized_start = out_len
        out.append(seq)
        out_len += len(seq)
        realized_end = out_len

        resolved_part = RealizedPart(
            name=part.name,
            role=part.role,
            kind=part.placement.kind,
            sequence_source=part.sequence.source,
            sequence_field=str(part.sequence.field) if part.sequence.field is not None else None,
            orientation=part.placement.orientation,
            start=site.start,
            end=site.end,
            sequence=seq,
            realized_start=realized_start,
            realized_end=realized_end,
        )
        realized[part.name] = resolved_part
        realized_ordered.append(resolved_part)
        cursor = site.end

    out.append(template_seq[cursor:])
    return AssembledConstruct(
        sequence="".join(out),
        ordered_parts=realized_ordered,
        parts_by_name=realized,
    )


def part_sequence(part: PartConfig, row: dict[str, object]) -> str:
    if part.sequence.source == "literal":
        seq = ensure_dna_text(str(part.sequence.literal), label=f"literal for part '{part.name}'")
    else:
        raw = row.get(str(part.sequence.field))
        if raw is None:
            raise ValidationError(
                f"Input row '{row.get('id')}' is missing field '{part.sequence.field}' for part '{part.name}'."
            )
        seq = ensure_dna_text(str(raw), label=f"input field '{part.sequence.field}' for part '{part.name}'")
    if part.placement.orientation == "reverse_complement":
        return reverse_complement(seq).upper()
    return seq


def extract_output_sequence(
    *,
    full_construct: str,
    realized_parts: dict[str, RealizedPart],
    cfg: JobConfig,
) -> tuple[str, int, int]:
    if cfg.job.realize is None or cfg.job.template is None:
        raise ValidationError("job.template and job.realize are required before extracting construct output.")
    realize = cfg.job.realize
    template = cfg.job.template
    if realize.mode == "full_construct":
        return full_construct, 0, len(full_construct)

    window = realize.window
    if window is None:
        raise ValidationError("realize.window must resolve before runtime extraction.")
    if realize.focal_part is None or realize.focal_part not in realized_parts:
        raise ValidationError(f"realize.focal_part '{realize.focal_part}' was not realized.")
    focal = realized_parts[realize.focal_part]
    geometry = normalize_window_geometry(
        full_construct_length=len(full_construct),
        template_circular=template.circular,
        focal=focal,
        window=window,
    )
    if template.circular:
        seq = "".join(
            full_construct[(geometry.start_raw + idx) % len(full_construct)] for idx in range(geometry.span_bp)
        )
        return seq, geometry.start, geometry.end
    return full_construct[geometry.start : geometry.end], geometry.start, geometry.end


def resolve_anchor_part(
    *,
    realized_parts: dict[str, RealizedPart],
    ordered_realized_parts: list[RealizedPart],
    focal_part_name: str | None,
) -> RealizedPart | None:
    if focal_part_name:
        candidate = realized_parts.get(focal_part_name)
        if candidate is not None:
            return candidate
    for part in ordered_realized_parts:
        if part.role == "anchor" or part.name == "anchor":
            return part
    return None
