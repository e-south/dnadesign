"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/scanning.py

Compatibility wrapper around the shared nickase scanning seam.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.nickases.scanning import EvaluatedMatch, display_motif_for_orientation
from dnadesign.cruncher.nickases.scanning import derive_nick_event as _derive_nick_event
from dnadesign.cruncher.nickases.scanning import enumerate_site_instances as _enumerate_site_instances


def derive_nick_event(*, cassette_offset: int, **kwargs):
    return _derive_nick_event(coordinate_offset=cassette_offset, **kwargs)


def enumerate_site_instances(sequence: str, *, cassette_offset: int, entry):
    return _enumerate_site_instances(sequence, coordinate_offset=cassette_offset, entry=entry)


__all__ = [
    "EvaluatedMatch",
    "derive_nick_event",
    "display_motif_for_orientation",
    "enumerate_site_instances",
]
