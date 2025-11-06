"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/layout.py
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Tuple

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


def _is_sigma(a: Annotation) -> bool:
    """True for any σ70 piece (plugin tag 'sigma' or dataset 'tf:sigma70_*')."""
    tag = (a.tag or "").lower()
    return tag == "sigma" or tag.startswith("tf:sigma70_")


def assign_tracks_forward_with_sigma_lock(annotations: List[Annotation]) -> List[int]:
    """
    Forward-strand track assignment with **σ70 lock**:
      - All σ70 annotations (−35, −10) are forced to **track 0**.
      - Non-σ annotations are assigned greedily to **tracks ≥ 1** (never track 0).
    This guarantees σ boxes share one plane and the spacer link can be drawn on it
    without other TFs overlapping that horizontal line.
    """
    if not annotations:
        return []

    # Partition forward annotations into sigma vs non-sigma.
    # (Render code calls this only for forward annotations.)
    sigma_idxs: List[int] = []
    other_idxs: List[int] = []
    for i, a in enumerate(annotations):
        (sigma_idxs if _is_sigma(a) else other_idxs).append(i)

    # Start with everyone unassigned; then stamp σ → track 0.
    tracks: List[int] = [-1] * len(annotations)
    for i in sigma_idxs:
        tracks[i] = 0

    # Build intervals currently occupying each track.
    # Track 0 has σ boxes; higher tracks will be filled greedily.
    track_ends: List[int] = []
    # Ensure list long enough to reference track 0.
    track_ends.append(
        max((annotations[i].end() for i in sigma_idxs), default=0)
    )  # track 0 terminal end (not really used for routing others)

    # Prepare non-sigma events with stable placement: priority by start, longer first.
    events: List[Tuple[int, int, int, int]] = []
    for i in other_idxs:
        a = annotations[i]
        events.append((a.start, a.end(), -(a.length), i))
    events.sort(key=lambda t: (t[0], t[1], t[2]))  # start, end, longer first

    # Greedy fit for non-σ annotations into tracks >= 1.
    for _, start, _, idx in events:
        placed = False
        # Try existing tracks **from 1 upward** (track 0 is reserved).
        for t in range(1, len(track_ends)):
            if track_ends[t] <= start:
                track_ends[t] = annotations[idx].end()
                tracks[idx] = t
                placed = True
                break
        if not placed:
            # Open a new track above current highest.
            track_ends.append(annotations[idx].end())
            tracks[idx] = len(track_ends) - 1

    return tracks
