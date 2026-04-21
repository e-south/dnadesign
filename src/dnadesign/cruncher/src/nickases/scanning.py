"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/nickases/scanning.py

Shared recognition-site scanning helpers for explicit nickase-aware workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.cruncher.nickases.models import (
    NickaseCatalogEntry,
    NickEvent,
    RecognitionSiteInstance,
)
from dnadesign.cruncher.nickases.scan_plan import build_entry_scan_plans, window_matches_plan


@dataclass(frozen=True)
class EvaluatedMatch:
    variant: NickaseCatalogEntry
    site: RecognitionSiteInstance
    nick: NickEvent

    def key(self) -> tuple[str, int, int, str, str, int]:
        return (
            self.variant.id,
            self.site.start,
            self.site.end,
            self.site.orientation,
            self.nick.strand,
            self.nick.boundary_context,
        )


def derive_nick_event(
    *,
    entry: NickaseCatalogEntry,
    start: int,
    orientation: str,
    coordinate_offset: int,
    motif_len: int,
) -> NickEvent:
    if orientation == "forward":
        if entry.top_cut_offset is not None:
            strand = "primary"
            boundary_context = start + entry.top_cut_offset
        else:
            strand = "complement"
            boundary_context = start + int(entry.bottom_cut_offset)
    else:
        if entry.top_cut_offset is not None:
            strand = "complement"
            boundary_context = start + (motif_len - entry.top_cut_offset)
        else:
            strand = "primary"
            boundary_context = start + (motif_len - int(entry.bottom_cut_offset))
    return NickEvent(
        variant_id=entry.id,
        specificity_id=entry.specificity_id,
        strand=strand,
        boundary=boundary_context - coordinate_offset,
        boundary_context=boundary_context,
        source_site_start=start,
        source_site_end=start + motif_len,
        source_site_orientation=orientation,
    )


def _raw_nick_geometry(
    *,
    entry: NickaseCatalogEntry,
    start: int,
    orientation: str,
    motif_len: int,
) -> tuple[str, int]:
    if orientation == "forward":
        if entry.top_cut_offset is not None:
            return "primary", start + entry.top_cut_offset
        return "complement", start + int(entry.bottom_cut_offset)
    if entry.top_cut_offset is not None:
        return "complement", start + (motif_len - entry.top_cut_offset)
    return "primary", start + (motif_len - int(entry.bottom_cut_offset))


def display_motif_for_orientation(entry: NickaseCatalogEntry, *, orientation: str) -> str:
    for plan in build_entry_scan_plans(entry):
        if plan.orientation == orientation:
            return plan.motif_text
    raise ValueError(f"Unsupported orientation {orientation!r} for nickase {entry.id}.")


def suffix_sensitive_scan_start(entry: NickaseCatalogEntry, *, prefix_length: int) -> int:
    plans = build_entry_scan_plans(entry)
    if not plans:
        return max(0, prefix_length)
    return max(0, prefix_length - plans[0].motif_len + 1)


def enumerate_boundary_placements(
    entry: NickaseCatalogEntry,
    *,
    boundary: int,
    required_strand: str | None = None,
) -> list[tuple[str, int]]:
    placements: list[tuple[str, int]] = []
    for plan in build_entry_scan_plans(entry):
        strand, boundary_offset = _raw_nick_geometry(
            entry=entry,
            start=0,
            orientation=plan.orientation,
            motif_len=plan.motif_len,
        )
        if required_strand is not None and strand != required_strand:
            continue
        placements.append((plan.orientation, boundary - boundary_offset))
    return placements


def build_evaluated_match(
    *,
    entry: NickaseCatalogEntry,
    start: int,
    orientation: str,
    coordinate_offset: int,
    matched_span_sequence: str,
) -> EvaluatedMatch:
    motif_len = entry.motif_len or len(entry.motif_top_5to3)
    strand, boundary_context = _raw_nick_geometry(
        entry=entry,
        start=start,
        orientation=orientation,
        motif_len=motif_len,
    )
    return EvaluatedMatch(
        variant=entry,
        site=RecognitionSiteInstance(
            variant_id=entry.id,
            specificity_id=entry.specificity_id,
            start=start,
            end=start + motif_len,
            orientation=orientation,
            matched_span_sequence=matched_span_sequence,
            local_start=start - coordinate_offset,
            local_end=start + motif_len - coordinate_offset,
        ),
        nick=NickEvent(
            variant_id=entry.id,
            specificity_id=entry.specificity_id,
            strand=strand,
            boundary=boundary_context - coordinate_offset,
            boundary_context=boundary_context,
            source_site_start=start,
            source_site_end=start + motif_len,
            source_site_orientation=orientation,
        ),
    )


def enumerate_site_instances(
    sequence: str,
    *,
    coordinate_offset: int,
    entry: NickaseCatalogEntry,
) -> list[EvaluatedMatch]:
    return enumerate_site_instances_starting_at_or_after(
        sequence,
        coordinate_offset=coordinate_offset,
        entry=entry,
        start_min=0,
    )


def enumerate_site_instances_starting_at_or_after(
    sequence: str,
    *,
    coordinate_offset: int,
    entry: NickaseCatalogEntry,
    start_min: int,
) -> list[EvaluatedMatch]:
    plans = build_entry_scan_plans(entry)
    if not plans:
        return []
    motif_len = plans[0].motif_len
    matches: list[EvaluatedMatch] = []
    window_limit = len(sequence) - motif_len
    if window_limit < 0:
        return matches
    for start in range(max(0, start_min), window_limit + 1):
        window = sequence[start : start + motif_len]
        for plan in plans:
            if not window_matches_plan(sequence, start=start, plan=plan):
                continue
            _strand, boundary_context = _raw_nick_geometry(
                entry=entry,
                start=start,
                orientation=plan.orientation,
                motif_len=plan.motif_len,
            )
            if boundary_context < 0 or boundary_context > len(sequence):
                continue
            matches.append(
                build_evaluated_match(
                    entry=entry,
                    start=start,
                    orientation=plan.orientation,
                    coordinate_offset=coordinate_offset,
                    matched_span_sequence=window,
                )
            )
    return matches
