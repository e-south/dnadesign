"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/realization/placement_guards.py

Guard extraction and validation for construct placement contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..contracts.config import PartConfig
from ..contracts.errors import ValidationError
from .placement_models import PlacementSite, TemplateLike
from .placement_search import template_context_sequence, template_match_offsets
from .sequences import ensure_dna_text


def validate_placement_guards(
    *,
    template: TemplateLike,
    part: PartConfig,
    site: PlacementSite,
) -> None:
    replaced_sequence = guard_replaced_sequence(part)
    template_interval = template.sequence[site.start : site.end]
    if replaced_sequence is not None and template_interval.upper() != replaced_sequence.upper():
        raise ValidationError(
            f"Part '{part.name}' expected template interval [{site.start}, {site.end}) to match "
            "placement.guards.replaced_sequence."
        )
    _require_unique_template_match(
        template=template,
        part=part,
        field_name="placement.guards.replaced_sequence",
        expected=replaced_sequence,
        aligned_start=site.start,
    )
    replaced_span_bp = guard_replaced_span_bp(part)
    if replaced_span_bp is not None and (site.end - site.start) != replaced_span_bp:
        raise ValidationError(
            f"Part '{part.name}' expected resolved replacement span {replaced_span_bp} bp, "
            f"but locator resolved {site.end - site.start} bp."
        )
    expected_upstream = guard_upstream_sequence(part)
    if expected_upstream is not None:
        observed_upstream = template_context_sequence(
            template.sequence,
            anchor=site.start,
            length=len(expected_upstream),
            circular=template.circular,
            direction="upstream",
        )
        if observed_upstream.upper() != expected_upstream.upper():
            raise ValidationError(
                f"Part '{part.name}' expected the forward-strand upstream flank ending at "
                f"{site.start} to match placement.guards.upstream_sequence."
            )
        _require_unique_template_match(
            template=template,
            part=part,
            field_name="placement.guards.upstream_sequence",
            expected=expected_upstream,
            aligned_start=site.start - len(expected_upstream),
        )
    expected_downstream = guard_downstream_sequence(part)
    if expected_downstream is not None:
        observed_downstream = template_context_sequence(
            template.sequence,
            anchor=site.end,
            length=len(expected_downstream),
            circular=template.circular,
            direction="downstream",
        )
        if observed_downstream.upper() != expected_downstream.upper():
            raise ValidationError(
                f"Part '{part.name}' expected the forward-strand downstream flank starting at "
                f"{site.end} to match placement.guards.downstream_sequence."
            )
        _require_unique_template_match(
            template=template,
            part=part,
            field_name="placement.guards.downstream_sequence",
            expected=expected_downstream,
            aligned_start=site.end,
        )


def guard_replaced_sequence(part: PartConfig) -> str | None:
    guards = part.placement.guards
    if guards is None or guards.replaced_sequence is None:
        return None
    return ensure_dna_text(
        str(guards.replaced_sequence),
        label=f"placement.guards.replaced_sequence for part '{part.name}'",
    )


def guard_upstream_sequence(part: PartConfig) -> str | None:
    guards = part.placement.guards
    if guards is None or guards.upstream_sequence is None:
        return None
    return ensure_dna_text(
        str(guards.upstream_sequence),
        label=f"placement.guards.upstream_sequence for part '{part.name}'",
    )


def guard_downstream_sequence(part: PartConfig) -> str | None:
    guards = part.placement.guards
    if guards is None or guards.downstream_sequence is None:
        return None
    return ensure_dna_text(
        str(guards.downstream_sequence),
        label=f"placement.guards.downstream_sequence for part '{part.name}'",
    )


def guard_replaced_span_bp(part: PartConfig) -> int | None:
    guards = part.placement.guards
    if guards is None or guards.replaced_span_bp is None:
        return None
    return int(guards.replaced_span_bp)


def guard_requires_unique_forward_matches(part: PartConfig) -> bool:
    guards = part.placement.guards
    return bool(guards is not None and guards.require_unique_forward_matches)


def placement_guard_mode(part: PartConfig) -> str:
    has_replaced_sequence = guard_replaced_sequence(part) is not None
    has_upstream = guard_upstream_sequence(part) is not None
    has_downstream = guard_downstream_sequence(part) is not None
    has_span = guard_replaced_span_bp(part) is not None
    if has_replaced_sequence and (has_upstream or has_downstream or has_span):
        return "replaced_sequence_and_context"
    if has_replaced_sequence:
        return "replaced_sequence"
    if has_upstream or has_downstream:
        return "context"
    if has_span:
        return "span"
    return "none"


def observed_guard_upstream_sequence(
    *,
    template: TemplateLike,
    part: PartConfig,
    site: PlacementSite,
) -> str | None:
    expected = guard_upstream_sequence(part)
    if expected is None:
        return None
    return template_context_sequence(
        template.sequence,
        anchor=site.start,
        length=len(expected),
        circular=template.circular,
        direction="upstream",
    )


def observed_guard_downstream_sequence(
    *,
    template: TemplateLike,
    part: PartConfig,
    site: PlacementSite,
) -> str | None:
    expected = guard_downstream_sequence(part)
    if expected is None:
        return None
    return template_context_sequence(
        template.sequence,
        anchor=site.end,
        length=len(expected),
        circular=template.circular,
        direction="downstream",
    )


def _require_unique_template_match(
    *,
    template: TemplateLike,
    part: PartConfig,
    field_name: str,
    expected: str | None,
    aligned_start: int,
) -> None:
    if not guard_requires_unique_forward_matches(part) or expected is None:
        return
    offsets = template_match_offsets(template.sequence, expected, circular=template.circular)
    if len(offsets) != 1:
        raise ValidationError(
            f"Part '{part.name}' requires a unique forward-strand match for {field_name}, "
            f"but found {len(offsets)} matches in template '{template.id}'. Use a longer kmer or "
            "disable placement.guards.require_unique_forward_matches explicitly."
        )
    expected_start = aligned_start % len(template.sequence) if template.circular else aligned_start
    if offsets[0] != expected_start:
        raise ValidationError(
            f"Part '{part.name}' requires {field_name} to anchor the configured placement uniquely, "
            f"but the only forward-strand match starts at template offset {offsets[0]} instead of "
            f"{expected_start}."
        )
