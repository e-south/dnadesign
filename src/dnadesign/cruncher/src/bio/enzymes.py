"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/bio/enzymes.py

Shared recognition-site validation and duplex-cut geometry helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.cruncher.bio.iupac import motif_matches, normalize_iupac, reverse_complement_iupac


@dataclass(frozen=True)
class CutGeometry:
    recognition_start: int
    recognition_end: int
    orientation: str
    top_boundary: int | None
    bottom_boundary: int | None
    overhang_sequence: str
    overhang_length: int
    protruding_strand: str | None


def recognition_matches_at(
    sequence: str,
    *,
    start: int,
    recognition_sequence: str,
    orientation: str,
) -> bool:
    motif = normalize_iupac(recognition_sequence)
    end = start + len(motif)
    if start < 0 or end > len(sequence):
        return False
    window = sequence[start:end]
    target = motif if orientation == "forward" else reverse_complement_iupac(motif)
    return motif_matches(window, target)


def _oriented_boundary(
    *,
    start: int,
    motif_len: int,
    offset: int | None,
    orientation: str,
) -> int | None:
    if offset is None:
        return None
    if orientation == "forward":
        return start + offset
    # Reverse-oriented sites swap the visible strand roles, so the top-strand
    # boundary uses the bottom-strand offset and vice versa.
    return start + (motif_len - offset)


def derive_cut_geometry(
    sequence: str,
    *,
    start: int,
    recognition_sequence: str,
    orientation: str,
    top_cut_offset: int | None,
    bottom_cut_offset: int | None,
) -> CutGeometry:
    motif = normalize_iupac(recognition_sequence)
    if orientation not in {"forward", "reverse"}:
        raise ValueError(f"Unknown site orientation: {orientation!r}")
    if not recognition_matches_at(
        sequence,
        start=start,
        recognition_sequence=motif,
        orientation=orientation,
    ):
        raise ValueError("recognition sequence does not match the requested site span")
    motif_len = len(motif)
    top_boundary = _oriented_boundary(
        start=start,
        motif_len=motif_len,
        offset=top_cut_offset if orientation == "forward" else bottom_cut_offset,
        orientation=orientation,
    )
    bottom_boundary = _oriented_boundary(
        start=start,
        motif_len=motif_len,
        offset=bottom_cut_offset if orientation == "forward" else top_cut_offset,
        orientation=orientation,
    )
    for label, boundary in (("top", top_boundary), ("bottom", bottom_boundary)):
        if boundary is None:
            continue
        if boundary < 0 or boundary > len(sequence):
            raise ValueError(f"{label} cut boundary is outside the source sequence")
    overhang_sequence = ""
    overhang_length = 0
    protruding_strand: str | None = None
    if top_boundary is not None and bottom_boundary is not None and top_boundary != bottom_boundary:
        left = min(top_boundary, bottom_boundary)
        right = max(top_boundary, bottom_boundary)
        overhang_length = right - left
        window = sequence[left:right]
        if top_boundary < bottom_boundary:
            overhang_sequence = window
            protruding_strand = "primary"
        else:
            overhang_sequence = reverse_complement_iupac(window)
            protruding_strand = "complement"
    return CutGeometry(
        recognition_start=start,
        recognition_end=start + motif_len,
        orientation=orientation,
        top_boundary=top_boundary,
        bottom_boundary=bottom_boundary,
        overhang_sequence=overhang_sequence,
        overhang_length=overhang_length,
        protruding_strand=protruding_strand,
    )
