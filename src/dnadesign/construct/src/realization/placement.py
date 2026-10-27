"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/realization/placement.py

Placement resolution and guard contracts for construct realization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable

from ..contracts.config import CoordinatePlacementLocatorConfig, FlankPlacementLocatorConfig, PartConfig
from ..contracts.errors import ValidationError
from .placement_guards import (
    guard_downstream_sequence,
    guard_replaced_sequence,
    guard_replaced_span_bp,
    guard_requires_unique_forward_matches,
    guard_upstream_sequence,
    observed_guard_downstream_sequence,
    observed_guard_upstream_sequence,
    placement_guard_mode,
    validate_placement_guards,
)
from .placement_models import PlacementPlan, PlacementSite, PlannedPlacement, TemplateLike
from .placement_search import template_context_sequence, template_match_offsets
from .sequences import ensure_dna_text

__all__ = [
    "PlannedPlacement",
    "PlacementPlan",
    "PlacementSite",
    "TemplateLike",
    "planned_placements",
    "resolved_placement_sites",
    "template_context_sequence",
    "template_match_offsets",
    "validate_placement_guards",
    "validate_placements",
]


def resolved_placement_sites(
    template: TemplateLike,
    parts: Iterable[PartConfig],
) -> dict[str, PlacementSite]:
    resolved: dict[str, PlacementSite] = {}
    for part in parts:
        if part.name in resolved:
            raise ValidationError(f"Duplicate part name '{part.name}'.")
        resolved[part.name] = _resolve_locator_site(template=template, part=part)
    return resolved


def validate_placements(
    template_len: int,
    parts: Iterable[PartConfig],
    *,
    resolved_sites: dict[str, PlacementSite],
) -> list[PlacementPlan]:
    indexed_parts = list(enumerate(parts))
    ordered = [
        PlacementPlan(part=part, site=resolved_sites[part.name])
        for _, part in sorted(
            indexed_parts,
            key=lambda item: (resolved_sites[item[1].name].start, item[0]),
        )
    ]
    prior_end = -1
    prior_name = None
    prior_start = None
    prior_template_end = None
    for resolved in ordered:
        start = resolved.site.start
        end = resolved.site.end
        if end > template_len:
            raise ValidationError(
                f"Part '{resolved.part.name}' placement end {end} exceeds template length {template_len}."
            )
        if prior_start is not None and start == prior_start and end != prior_template_end:
            raise ValidationError(
                f"Part '{resolved.part.name}' shares template start {start} with part '{prior_name}' "
                "but uses a different "
                "template end. Same-start placements with different intervals are ambiguous; use distinct start "
                "coordinates or split them into separate construct jobs."
            )
        if start < prior_end:
            raise ValidationError(
                f"Part '{resolved.part.name}' overlaps prior placement '{prior_name}'. Placements must not overlap."
            )
        prior_end = end
        prior_name = resolved.part.name
        prior_start = start
        prior_template_end = end
    return ordered


def planned_placements(
    parts: Iterable[PartConfig],
    *,
    template: TemplateLike,
    resolved_sites: dict[str, PlacementSite],
) -> list[PlannedPlacement]:
    return [
        PlannedPlacement(
            part_name=part.name,
            part_role=part.role,
            sequence_source=part.sequence.source,
            sequence_field=str(part.sequence.field) if part.sequence.field is not None else None,
            placement_kind=part.placement.kind,
            template_start=resolved_sites[part.name].start,
            template_end=resolved_sites[part.name].end,
            template_span_bp=resolved_sites[part.name].end - resolved_sites[part.name].start,
            orientation=part.placement.orientation,
            locator_kind=resolved_sites[part.name].locator_kind,
            locator_upstream_sequence=resolved_sites[part.name].locator_upstream_sequence,
            locator_downstream_sequence=resolved_sites[part.name].locator_downstream_sequence,
            guard_mode=placement_guard_mode(part),
            guard_require_unique_forward_matches=guard_requires_unique_forward_matches(part),
            guard_replaced_span_bp=guard_replaced_span_bp(part),
            template_sequence=template.sequence[resolved_sites[part.name].start : resolved_sites[part.name].end],
            guard_replaced_sequence=guard_replaced_sequence(part),
            guard_upstream_sequence=guard_upstream_sequence(part),
            observed_guard_upstream_sequence=observed_guard_upstream_sequence(
                template=template,
                part=part,
                site=resolved_sites[part.name],
            ),
            guard_downstream_sequence=guard_downstream_sequence(part),
            observed_guard_downstream_sequence=observed_guard_downstream_sequence(
                template=template,
                part=part,
                site=resolved_sites[part.name],
            ),
        )
        for part in parts
    ]


def _resolve_locator_site(
    *,
    template: TemplateLike,
    part: PartConfig,
) -> PlacementSite:
    locator = part.placement.locator
    if isinstance(locator, CoordinatePlacementLocatorConfig):
        return PlacementSite(
            start=locator.start,
            end=locator.end,
            locator_kind="coordinates",
            locator_upstream_sequence=None,
            locator_downstream_sequence=None,
        )

    upstream = _locator_upstream_sequence(part)
    downstream = _locator_downstream_sequence(part)
    if upstream is None or downstream is None:
        raise ValidationError(f"Part '{part.name}' flank locator could not be normalized.")
    upstream_offsets = template_match_offsets(template.sequence, upstream, circular=template.circular)
    downstream_offsets = template_match_offsets(template.sequence, downstream, circular=template.circular)
    if len(upstream_offsets) != 1:
        raise ValidationError(
            f"Part '{part.name}' flank locator requires exactly one forward-strand match for "
            f"placement.locator.upstream_sequence, but found {len(upstream_offsets)} matches in template "
            f"'{template.id}'. Use a longer flank or fall back to coordinates."
        )
    if len(downstream_offsets) != 1:
        raise ValidationError(
            f"Part '{part.name}' flank locator requires exactly one forward-strand match for "
            f"placement.locator.downstream_sequence, but found {len(downstream_offsets)} matches in template "
            f"'{template.id}'. Use a longer flank or fall back to coordinates."
        )
    start = upstream_offsets[0] + len(upstream)
    end = downstream_offsets[0]
    if end < start:
        raise ValidationError(
            f"Part '{part.name}' flank locator resolves across the template origin or into overlapping flanks "
            f"(upstream_end={start}, downstream_start={end}). Explicit wraparound flank placement is not supported; "
            "provide coordinates instead."
        )
    if part.placement.kind == "replace" and end == start:
        raise ValidationError(
            f"Part '{part.name}' flank locator resolves to a zero-length interval. Use kind='insert' for a pure "
            "boundary insertion or widen the flanks to bracket a replace span."
        )
    if part.placement.kind == "insert" and end != start:
        raise ValidationError(
            f"Part '{part.name}' kind='insert' requires adjacent flanks, but the flank locator resolves to "
            f"{end - start} bp between the matches. Use kind='replace' or provide adjacent flanks."
        )
    return PlacementSite(
        start=start,
        end=end,
        locator_kind="flanks",
        locator_upstream_sequence=upstream,
        locator_downstream_sequence=downstream,
    )


def _locator_upstream_sequence(part: PartConfig) -> str | None:
    locator = part.placement.locator
    if not isinstance(locator, FlankPlacementLocatorConfig):
        return None
    return ensure_dna_text(
        str(locator.upstream_sequence),
        label=f"placement.locator.upstream_sequence for part '{part.name}'",
    )


def _locator_downstream_sequence(part: PartConfig) -> str | None:
    locator = part.placement.locator
    if not isinstance(locator, FlankPlacementLocatorConfig):
        return None
    return ensure_dna_text(
        str(locator.downstream_sequence),
        label=f"placement.locator.downstream_sequence for part '{part.name}'",
    )
