"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/release_enzymes/scanning.py

Recognition-site scanning helpers for release enzymes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from dnadesign.cruncher.nickases.models import (
    iupac_bases_for_symbol,
    normalize_iupac,
    reverse_complement_iupac,
)
from dnadesign.cruncher.release_enzymes.models import (
    ReleaseCutEvent,
    ReleaseEnzymeEntry,
    ReleaseRecognitionSiteInstance,
)


@dataclass(frozen=True)
class ReleaseOrientationScanPlan:
    orientation: Literal["forward", "reverse"]
    motif_text: str
    motif_len: int
    allowed_bases_by_position: tuple[frozenset[str], ...]


@dataclass(frozen=True)
class ReleaseEvaluatedMatch:
    variant: ReleaseEnzymeEntry
    site: ReleaseRecognitionSiteInstance
    cut: ReleaseCutEvent

    def key(self) -> tuple[str, int, int, str, int, int]:
        return (
            self.variant.variant_id,
            self.site.start,
            self.site.end,
            self.site.orientation,
            self.cut.top_cut_boundary,
            self.cut.bottom_cut_boundary,
        )


def _allowed_bases_by_position(motif_text: str) -> tuple[frozenset[str], ...]:
    return tuple(frozenset(iupac_bases_for_symbol(symbol)) for symbol in motif_text)


@lru_cache(maxsize=256)
def _cached_scan_plans(recognition_sequence: str) -> tuple[ReleaseOrientationScanPlan, ...]:
    motif_text = normalize_iupac(recognition_sequence)
    reverse_text = reverse_complement_iupac(motif_text)
    plans = [
        ReleaseOrientationScanPlan(
            orientation="forward",
            motif_text=motif_text,
            motif_len=len(motif_text),
            allowed_bases_by_position=_allowed_bases_by_position(motif_text),
        )
    ]
    if reverse_text != motif_text:
        plans.append(
            ReleaseOrientationScanPlan(
                orientation="reverse",
                motif_text=reverse_text,
                motif_len=len(reverse_text),
                allowed_bases_by_position=_allowed_bases_by_position(reverse_text),
            )
        )
    return tuple(plans)


def build_entry_scan_plans(entry: ReleaseEnzymeEntry) -> tuple[ReleaseOrientationScanPlan, ...]:
    return _cached_scan_plans(entry.recognition_sequence)


def display_motif_for_orientation(entry: ReleaseEnzymeEntry, *, orientation: Literal["forward", "reverse"]) -> str:
    for plan in build_entry_scan_plans(entry):
        if plan.orientation == orientation:
            return plan.motif_text
    raise ValueError(f"Unsupported orientation {orientation!r} for release enzyme {entry.variant_id}.")


def derive_release_cut(
    *,
    entry: ReleaseEnzymeEntry,
    start: int,
    orientation: Literal["forward", "reverse"],
) -> ReleaseCutEvent:
    motif_len = entry.recognition_len
    if orientation == "forward":
        top_cut_boundary = start + entry.top_cut_offset
        bottom_cut_boundary = start + entry.bottom_cut_offset
    else:
        top_cut_boundary = start + (motif_len - entry.bottom_cut_offset)
        bottom_cut_boundary = start + (motif_len - entry.top_cut_offset)
    return ReleaseCutEvent(
        variant_id=entry.variant_id,
        top_cut_boundary=top_cut_boundary,
        bottom_cut_boundary=bottom_cut_boundary,
        source_site_start=start,
        source_site_end=start + motif_len,
        source_site_orientation=orientation,
    )


def build_evaluated_match(
    *,
    entry: ReleaseEnzymeEntry,
    start: int,
    orientation: Literal["forward", "reverse"],
    coordinate_offset: int,
    matched_span_sequence: str,
) -> ReleaseEvaluatedMatch:
    cut = derive_release_cut(entry=entry, start=start, orientation=orientation)
    return ReleaseEvaluatedMatch(
        variant=entry,
        site=ReleaseRecognitionSiteInstance(
            variant_id=entry.variant_id,
            start=start,
            end=start + entry.recognition_len,
            orientation=orientation,
            matched_span_sequence=matched_span_sequence,
            local_start=start - coordinate_offset,
            local_end=start + entry.recognition_len - coordinate_offset,
        ),
        cut=cut,
    )


def enumerate_top_cut_placements(
    entry: ReleaseEnzymeEntry,
    *,
    top_cut_boundary: int,
) -> list[tuple[Literal["forward", "reverse"], int]]:
    placements: list[tuple[Literal["forward", "reverse"], int]] = []
    for plan in build_entry_scan_plans(entry):
        cut = derive_release_cut(entry=entry, start=0, orientation=plan.orientation)
        placements.append((plan.orientation, top_cut_boundary - cut.top_cut_boundary))
    return placements


def window_matches_plan(sequence: str, *, start: int, plan: ReleaseOrientationScanPlan) -> bool:
    end = start + plan.motif_len
    if start < 0 or end > len(sequence):
        return False
    for offset, allowed in enumerate(plan.allowed_bases_by_position):
        if sequence[start + offset] not in allowed:
            return False
    return True


def enumerate_site_instances(
    sequence: str,
    *,
    coordinate_offset: int,
    entry: ReleaseEnzymeEntry,
) -> list[ReleaseEvaluatedMatch]:
    plans = build_entry_scan_plans(entry)
    if not plans:
        return []
    motif_len = plans[0].motif_len
    matches: list[ReleaseEvaluatedMatch] = []
    window_limit = len(sequence) - motif_len
    if window_limit < 0:
        return matches
    for start in range(window_limit + 1):
        window = sequence[start : start + motif_len]
        for plan in plans:
            if not window_matches_plan(sequence, start=start, plan=plan):
                continue
            cut = derive_release_cut(entry=entry, start=start, orientation=plan.orientation)
            if cut.top_cut_boundary < 0 or cut.bottom_cut_boundary < 0:
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


__all__ = [
    "ReleaseEvaluatedMatch",
    "ReleaseOrientationScanPlan",
    "build_entry_scan_plans",
    "build_evaluated_match",
    "derive_release_cut",
    "display_motif_for_orientation",
    "enumerate_site_instances",
    "enumerate_top_cut_placements",
]
