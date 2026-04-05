"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/bsmbi.py

Shared BsmBI helpers for payload-centric YIU rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.bio import derive_cut_geometry, reverse_complement_iupac
from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload

BSMBI_RECOGNITION_SEQUENCE = "CGTCTC"
BSMBI_PRIMARY_CUT_OFFSET = 7
BSMBI_COMPLEMENT_CUT_OFFSET = 11
BSMBI_GHOST_SPACER_BASE = "N"


class DisplaySpan(StrictBaseModel):
    start: int
    end: int
    coordinate_space: str


class GhostExcisedContext(StrictBaseModel):
    coordinate_space: str = "fragment_display_nt"
    primary_indices: list[int]
    complement_indices: list[int]


class SplitFragmentDisplaySpec(StrictBaseModel):
    fragment_side: str
    panel_order: int
    title: str
    display_primary_sequence_5to3: str
    display_complement_sequence_3to5: str
    retained_primary_sequence_5to3: str
    retained_complement_sequence_3to5: str
    retained_payload_body_sequence_5to3: str
    selected_sticky_end_sequence_5to3: str
    canonical_sticky_end_sequence_5to3: str
    sticky_end_display_span: DisplaySpan
    payload_body_display_span: DisplaySpan
    retained_primary_display_span: DisplaySpan
    retained_complement_display_span: DisplaySpan
    payload_junction_window: DisplaySpan
    sticky_end_orientation: str
    recognition_site_orientation: str
    ghost_excised_context: GhostExcisedContext | None = None


def aligned_complement_3to5(sequence_5to3: str) -> str:
    return "".join(reverse_complement_iupac(base) for base in sequence_5to3)


def assembled_payload_aligned_complement_3to5(normalized: NormalizedPayload) -> str:
    return normalized.selected_complement_sequence


def _fragment_display_span(*, start: int, end: int) -> DisplaySpan:
    return DisplaySpan(start=start, end=end, coordinate_space="fragment_display_nt")


def _payload_span(normalized: NormalizedPayload) -> DisplaySpan:
    return DisplaySpan(start=normalized.junction.start, end=normalized.junction.end, coordinate_space="payload_forward")


def _build_forward_fragment_display(
    *,
    fragment_side: str,
    panel_order: int,
    title: str,
    sticky_end_sequence_5to3: str,
    sticky_end_payload_sequence_3to5: str,
    canonical_sticky_end_sequence_5to3: str,
    payload_body_sequence_5to3: str,
    payload_body_sequence_3to5: str,
    payload_junction_window: DisplaySpan,
) -> SplitFragmentDisplaySpec:
    display_primary = (
        f"{BSMBI_RECOGNITION_SEQUENCE}{BSMBI_GHOST_SPACER_BASE}{sticky_end_sequence_5to3}{payload_body_sequence_5to3}"
    )
    display_complement = (
        aligned_complement_3to5(f"{BSMBI_RECOGNITION_SEQUENCE}{BSMBI_GHOST_SPACER_BASE}")
        + sticky_end_payload_sequence_3to5
        + payload_body_sequence_3to5
    )
    geometry = derive_cut_geometry(
        display_primary,
        start=0,
        recognition_sequence=BSMBI_RECOGNITION_SEQUENCE,
        orientation="forward",
        top_cut_offset=BSMBI_PRIMARY_CUT_OFFSET,
        bottom_cut_offset=BSMBI_COMPLEMENT_CUT_OFFSET,
    )
    assert geometry.top_boundary is not None
    assert geometry.bottom_boundary is not None
    return SplitFragmentDisplaySpec(
        fragment_side=fragment_side,
        panel_order=panel_order,
        title=title,
        display_primary_sequence_5to3=display_primary,
        display_complement_sequence_3to5=display_complement,
        retained_primary_sequence_5to3=display_primary[geometry.top_boundary :],
        retained_complement_sequence_3to5=display_complement[geometry.bottom_boundary :],
        retained_payload_body_sequence_5to3=payload_body_sequence_5to3,
        selected_sticky_end_sequence_5to3=sticky_end_sequence_5to3,
        canonical_sticky_end_sequence_5to3=canonical_sticky_end_sequence_5to3,
        sticky_end_display_span=_fragment_display_span(start=geometry.top_boundary, end=geometry.bottom_boundary),
        payload_body_display_span=_fragment_display_span(start=geometry.bottom_boundary, end=len(display_primary)),
        retained_primary_display_span=_fragment_display_span(start=geometry.top_boundary, end=len(display_primary)),
        retained_complement_display_span=_fragment_display_span(
            start=geometry.bottom_boundary,
            end=len(display_complement),
        ),
        payload_junction_window=payload_junction_window,
        sticky_end_orientation="inward",
        recognition_site_orientation="inward",
        ghost_excised_context=GhostExcisedContext(
            primary_indices=list(range(0, geometry.top_boundary)),
            complement_indices=list(range(0, geometry.bottom_boundary)),
        ),
    )


def _build_reverse_fragment_display(
    *,
    fragment_side: str,
    panel_order: int,
    title: str,
    sticky_end_sequence_5to3: str,
    sticky_end_payload_sequence_3to5: str,
    canonical_sticky_end_sequence_5to3: str,
    payload_body_sequence_5to3: str,
    payload_body_sequence_3to5: str,
    payload_junction_window: DisplaySpan,
) -> SplitFragmentDisplaySpec:
    recognition_reverse_complement = reverse_complement_iupac(BSMBI_RECOGNITION_SEQUENCE)
    display_primary = (
        f"{payload_body_sequence_5to3}"
        f"{sticky_end_sequence_5to3}"
        f"{BSMBI_GHOST_SPACER_BASE}"
        f"{recognition_reverse_complement}"
    )
    display_complement = (
        f"{payload_body_sequence_3to5}"
        f"{sticky_end_payload_sequence_3to5}"
        f"{BSMBI_GHOST_SPACER_BASE}"
        f"{BSMBI_RECOGNITION_SEQUENCE[::-1]}"
    )
    recognition_start = len(display_primary) - len(recognition_reverse_complement)
    geometry = derive_cut_geometry(
        display_primary,
        start=recognition_start,
        recognition_sequence=BSMBI_RECOGNITION_SEQUENCE,
        orientation="reverse",
        top_cut_offset=BSMBI_PRIMARY_CUT_OFFSET,
        bottom_cut_offset=BSMBI_COMPLEMENT_CUT_OFFSET,
    )
    assert geometry.top_boundary is not None
    assert geometry.bottom_boundary is not None
    return SplitFragmentDisplaySpec(
        fragment_side=fragment_side,
        panel_order=panel_order,
        title=title,
        display_primary_sequence_5to3=display_primary,
        display_complement_sequence_3to5=display_complement,
        retained_primary_sequence_5to3=display_primary[: geometry.top_boundary],
        retained_complement_sequence_3to5=display_complement[: geometry.bottom_boundary],
        retained_payload_body_sequence_5to3=payload_body_sequence_5to3,
        selected_sticky_end_sequence_5to3=sticky_end_sequence_5to3,
        canonical_sticky_end_sequence_5to3=canonical_sticky_end_sequence_5to3,
        sticky_end_display_span=_fragment_display_span(start=geometry.top_boundary, end=geometry.bottom_boundary),
        payload_body_display_span=_fragment_display_span(start=0, end=len(payload_body_sequence_5to3)),
        retained_primary_display_span=_fragment_display_span(start=0, end=geometry.top_boundary),
        retained_complement_display_span=_fragment_display_span(start=0, end=geometry.bottom_boundary),
        payload_junction_window=payload_junction_window,
        sticky_end_orientation="inward",
        recognition_site_orientation="inward",
        ghost_excised_context=GhostExcisedContext(
            primary_indices=list(range(geometry.top_boundary, len(display_primary))),
            complement_indices=list(range(geometry.bottom_boundary, len(display_complement))),
        ),
    )


def build_split_fragment_display_specs(
    normalized: NormalizedPayload,
) -> tuple[SplitFragmentDisplaySpec, SplitFragmentDisplaySpec]:
    left_body = normalized.selected_payload_sequence[: normalized.junction.start]
    right_body = normalized.selected_payload_sequence[normalized.junction.end :]
    selected_sticky_end = normalized.selected_complement_sequence[normalized.junction.start : normalized.junction.end][
        ::-1
    ]
    selected_payload_sticky_end = normalized.selected_payload_sequence[
        normalized.junction.start : normalized.junction.end
    ][::-1]
    canonical_sticky_end = normalized.reference_complement_sequence[
        normalized.junction.start : normalized.junction.end
    ][::-1]
    payload_junction_window = _payload_span(normalized)
    left = _build_forward_fragment_display(
        fragment_side="left",
        panel_order=0,
        title="Split payload left",
        sticky_end_sequence_5to3=selected_sticky_end,
        sticky_end_payload_sequence_3to5=selected_payload_sticky_end,
        canonical_sticky_end_sequence_5to3=canonical_sticky_end,
        payload_body_sequence_5to3=reverse_complement_iupac(left_body),
        payload_body_sequence_3to5=left_body[::-1],
        payload_junction_window=payload_junction_window,
    )
    right = _build_reverse_fragment_display(
        fragment_side="right",
        panel_order=1,
        title="Split payload right",
        sticky_end_sequence_5to3=selected_sticky_end,
        sticky_end_payload_sequence_3to5=selected_payload_sticky_end,
        canonical_sticky_end_sequence_5to3=canonical_sticky_end,
        payload_body_sequence_5to3=reverse_complement_iupac(right_body),
        payload_body_sequence_3to5=right_body[::-1],
        payload_junction_window=payload_junction_window,
    )
    return left, right
