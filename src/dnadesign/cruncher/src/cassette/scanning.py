"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/scanning.py

Shared recognition-site scanning helpers for cassette workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.cruncher.cassette.models import (
    NickaseCatalogEntry,
    NickEvent,
    RecognitionSiteInstance,
    motif_matches,
    reverse_complement_iupac,
)


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
    cassette_offset: int,
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
        boundary=boundary_context - cassette_offset,
        boundary_context=boundary_context,
        source_site_start=start,
        source_site_end=start + motif_len,
        source_site_orientation=orientation,  # type: ignore[arg-type]
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
    motif = entry.motif_top_5to3
    return motif if orientation == "forward" else reverse_complement_iupac(motif)


def enumerate_site_instances(
    sequence: str,
    *,
    cassette_offset: int,
    entry: NickaseCatalogEntry,
) -> list[EvaluatedMatch]:
    motif = entry.motif_top_5to3
    motif_rc = reverse_complement_iupac(motif)
    motif_len = entry.motif_len or len(motif)
    matches: list[EvaluatedMatch] = []
    for start in range(0, len(sequence) - motif_len + 1):
        window = sequence[start : start + motif_len]
        orientations: list[str] = []
        if motif_matches(window, motif):
            orientations.append("forward")
        if motif_matches(window, motif_rc) and motif_rc != motif:
            orientations.append("reverse")
        for orientation in orientations:
            strand, boundary_context = _raw_nick_geometry(
                entry=entry,
                start=start,
                orientation=orientation,
                motif_len=motif_len,
            )
            if boundary_context < 0 or boundary_context > len(sequence):
                continue
            site = RecognitionSiteInstance(
                variant_id=entry.id,
                specificity_id=entry.specificity_id,
                start=start,
                end=start + motif_len,
                orientation=orientation,  # type: ignore[arg-type]
                matched_span_sequence=window,
                cassette_start=start - cassette_offset,
                cassette_end=start + motif_len - cassette_offset,
            )
            matches.append(
                EvaluatedMatch(
                    variant=entry,
                    site=site,
                    nick=NickEvent(
                        variant_id=entry.id,
                        specificity_id=entry.specificity_id,
                        strand=strand,  # type: ignore[arg-type]
                        boundary=boundary_context - cassette_offset,
                        boundary_context=boundary_context,
                        source_site_start=start,
                        source_site_end=start + motif_len,
                        source_site_orientation=orientation,  # type: ignore[arg-type]
                    ),
                )
            )
    return matches
