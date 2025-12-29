"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_tracks.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.baserender.src.contracts import BoundsError
from dnadesign.baserender.src.layout import assign_tracks_forward, assign_tracks_generic
from dnadesign.baserender.src.model import Annotation


def test_assign_tracks_forward_with_sigma_reserves_zero():
    anns = [
        Annotation(
            start=0,
            length=3,
            strand="fwd",
            label="AAA",
            tag="sigma",
            payload={"priority": 0},
        ),
        Annotation(
            start=10,
            length=3,
            strand="fwd",
            label="TTT",
            tag="tf:lexa",
        ),
    ]
    tracks = assign_tracks_forward(anns)
    assert tracks[0] == 0
    assert tracks[1] >= 1


def test_assign_tracks_forward_without_sigma_starts_at_zero():
    anns = [
        Annotation(
            start=0,
            length=3,
            strand="fwd",
            label="AAA",
            tag="tf:lexa",
        ),
        Annotation(
            start=10,
            length=3,
            strand="fwd",
            label="TTT",
            tag="tf:cpxr",
        ),
    ]
    tracks = assign_tracks_forward(anns)
    assert min(tracks) == 0


def _assert_no_overlap(annotations, tracks):
    by_track = {}
    for a, t in zip(annotations, tracks):
        by_track.setdefault(t, []).append((a.start, a.end()))
    for spans in by_track.values():
        spans = sorted(spans)
        for (s1, e1), (s2, _e2) in zip(spans, spans[1:]):
            assert s2 >= e1


def test_sigma_overlaps_raise():
    anns = [
        Annotation(
            start=0,
            length=4,
            strand="fwd",
            label="AAAA",
            tag="sigma",
            payload={"priority": 0},
        ),
        Annotation(
            start=2,
            length=4,
            strand="fwd",
            label="CCCC",
            tag="sigma",
            payload={"priority": 0},
        ),
    ]
    with pytest.raises(BoundsError):
        _ = assign_tracks_forward(anns)


def test_assign_tracks_generic_no_overlap():
    anns = [
        Annotation(start=0, length=4, strand="fwd", label="AAAA", tag="tf:a"),
        Annotation(start=2, length=4, strand="fwd", label="CCCC", tag="tf:b"),
        Annotation(start=6, length=3, strand="fwd", label="GGG", tag="tf:c"),
    ]
    tracks = assign_tracks_generic(anns)
    _assert_no_overlap(anns, tracks)
