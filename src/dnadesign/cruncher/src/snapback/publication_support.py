"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/publication_support.py

Shared helpers for snapback QA/public publication geometry.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.cruncher.nickases.models import reverse_complement


def complement_sequence(sequence: str) -> str:
    return reverse_complement(sequence)[::-1]


def protected_overlap_span(*, candidate) -> dict[str, int] | None:
    overlap_start = max(candidate.protected_region.start, candidate.retained_homology_window.start)
    overlap_end = min(candidate.protected_region.end, candidate.retained_homology_window.end)
    if overlap_end <= overlap_start:
        return None
    local_start = overlap_start - candidate.retained_homology_window.start
    local_end = overlap_end - candidate.retained_homology_window.start
    return {
        "start": candidate.post_nick_retained_homology_span.start + local_start,
        "end": candidate.post_nick_retained_homology_span.start + local_end,
    }


def absolute_primary_mismatch_positions(candidate) -> list[int]:
    return [candidate.post_nick_retained_homology_span.start + position for position in candidate.mismatch_positions]


def absolute_foldback_partner_mismatch_positions(candidate) -> list[int]:
    return [candidate.post_nick_foldback_arm_span.end - 1 - position for position in candidate.mismatch_positions]


def effective_cap_span(candidate) -> dict[str, int]:
    return {
        "start": candidate.source_cap_window.start,
        "end": candidate.cap_span.end,
    }


def foldback_loop_geometry(candidate) -> dict[str, Any]:
    return {
        "kind": "hairpin_corner_triloop_v1",
        "source_cap_span": candidate.post_nick_source_cap_span.model_dump(mode="json"),
        "cap_extension_span": candidate.post_nick_cap_extension_span.model_dump(mode="json"),
        "display_primary_span": candidate.post_nick_retained_homology_span.model_dump(mode="json"),
        "display_complement_span": candidate.post_nick_foldback_arm_span.model_dump(mode="json"),
    }
