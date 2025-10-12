"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/layout.py
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from .contracts import Size
from .model import Annotation

DNA_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def comp(seq: str) -> str:
    return seq.translate(DNA_COMP)


def revcomp(seq: str) -> str:
    return comp(seq)[::-1]


def measure_char_cell(font_family: str, font_size: int, dpi: int) -> Size:
    """
    Measure average monospace cell size from many glyphs (robust for mono fonts).
    """
    prop = FontProperties(family=font_family, size=font_size)
    N = 64
    run = "M" * N  # any mono glyph works; 'M' is reliably widest in mono sets
    bbox_w = TextPath((0, 0), run, prop=prop).get_extents()
    cw_pt = bbox_w.width / N
    # Height from 'Ag' captures ascenders/descenders without padding guesses
    ch_pt = TextPath((0, 0), "Ag", prop=prop).get_extents().height
    cw = cw_pt / 72.0 * dpi
    ch = ch_pt / 72.0 * dpi
    return Size(cw, ch)


def assign_tracks(annotations: List[Annotation]) -> List[int]:
    """
    Greedy interval coloring with **priority** (lower is closer to baseline).
    Each annotation may set payload['priority'] (int). Missing → 10.
    """

    def prio(a: Annotation) -> int:
        try:
            if a.payload is not None and "priority" in a.payload:
                return int(a.payload["priority"])  # type: ignore[arg-type]
        except Exception:
            pass
        return 10

    # Sort by (priority, start, longer first) to place important/longer boxes first.
    events = sorted(
        [(prio(a), a.start, a.start + a.length, i) for i, a in enumerate(annotations)],
        key=lambda x: (x[0], x[1], -(x[2] - x[1])),
    )

    tracks: List[int] = [-1] * len(annotations)
    track_ends: List[int] = []  # end position per track
    for _, start, end, idx in events:
        placed = False
        for t, last_end in enumerate(track_ends):
            if last_end <= start:
                track_ends[t] = end
                tracks[idx] = t
                placed = True
                break
        if not placed:
            track_ends.append(end)
            tracks[idx] = len(track_ends) - 1
    return tracks
