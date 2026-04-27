"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/visual/snapback_visual_v1.py

Shared snapback visual contract for nucleotide-resolution QA rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .common import CoordinateSpan, JsonMap, PositiveLengthSpan, VisualContractModel


class SnapbackPairingV1(VisualContractModel):
    left_index: int = Field(ge=0)
    right_index: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_order(self) -> "SnapbackPairingV1":
        if self.right_index <= self.left_index:
            raise ValueError("pairings right_index must be > left_index")
        return self


class SnapbackLoopGeometryV1(VisualContractModel):
    kind: Literal["hairpin_corner_triloop_v1"] = "hairpin_corner_triloop_v1"
    source_cap_span: CoordinateSpan
    cap_extension_span: CoordinateSpan
    display_primary_span: PositiveLengthSpan
    display_complement_span: PositiveLengthSpan

    @model_validator(mode="after")
    def _validate_cap_partition(self) -> "SnapbackLoopGeometryV1":
        if self.source_cap_span.end != self.cap_extension_span.start:
            raise ValueError("source_cap_span must end at cap_extension_span.start")
        return self


class SnapbackVisualV1(VisualContractModel):
    contract_kind: Literal["snapback_visual_v1"] = "snapback_visual_v1"
    state_id: str
    state_kind: Literal["pre_nick_duplex", "post_nick_exposed", "post_nick_foldback"]
    alphabet: Literal["dna", "iupac_dna"] = "dna"
    title: str | None = None
    primary_sequence: str
    complement_sequence: str
    primary_row_label: str
    complement_row_label: str
    nick_boundary: int | None = Field(default=None, ge=0)
    ligation_junction_boundary: int = Field(ge=0)
    protected_region_span: PositiveLengthSpan | None = None
    pre_nick_duplex_window_span: PositiveLengthSpan | None = None
    intended_site_span: PositiveLengthSpan | None = None
    anchored_duplex_span: CoordinateSpan | None = None
    released_prefix_span: CoordinateSpan | None = None
    retained_stem_span: PositiveLengthSpan
    released_suffix_span: CoordinateSpan | None = None
    cap_span: CoordinateSpan | None = None
    foldback_revcomp_span: PositiveLengthSpan
    exposed_complement_span: CoordinateSpan | None = None
    loop_geometry: SnapbackLoopGeometryV1 | None = None
    pairings: list[SnapbackPairingV1] = Field(default_factory=list)
    primary_mismatch_positions: list[int] = Field(default_factory=list)
    complement_mismatch_positions: list[int] = Field(default_factory=list)
    meta: JsonMap = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_contract(self) -> "SnapbackVisualV1":
        if not self.primary_sequence:
            raise ValueError("primary_sequence must be non-empty")
        if len(self.complement_sequence) != len(self.primary_sequence):
            raise ValueError("complement_sequence must match primary_sequence length")
        limit = len(self.primary_sequence)

        if self.ligation_junction_boundary != self.retained_stem_span.start:
            raise ValueError("ligation_junction_boundary must equal retained_stem_span.start")
        if self.ligation_junction_boundary > limit:
            raise ValueError("ligation_junction_boundary must lie within primary_sequence bounds")
        if self.nick_boundary is not None and self.nick_boundary > limit:
            raise ValueError("nick_boundary must lie within primary_sequence bounds")

        for name, span in (
            ("anchored_duplex_span", self.anchored_duplex_span),
            ("released_prefix_span", self.released_prefix_span),
            ("released_suffix_span", self.released_suffix_span),
            ("cap_span", self.cap_span),
            ("exposed_complement_span", self.exposed_complement_span),
        ):
            if span is not None and span.end > limit:
                raise ValueError(f"{name} must lie within primary_sequence bounds")
        for name, span in (
            ("protected_region_span", self.protected_region_span),
            ("pre_nick_duplex_window_span", self.pre_nick_duplex_window_span),
            ("intended_site_span", self.intended_site_span),
            ("retained_stem_span", self.retained_stem_span),
            ("foldback_revcomp_span", self.foldback_revcomp_span),
        ):
            if span is not None and span.end > limit:
                raise ValueError(f"{name} must lie within primary_sequence bounds")

        if self.state_kind in {"pre_nick_duplex", "post_nick_exposed"}:
            if self.nick_boundary is None:
                raise ValueError("nick_boundary is required for pre/exposed states")
            if self.ligation_junction_boundary != self.nick_boundary:
                raise ValueError("pre/exposed states must use the nick boundary as the snapback origin")
            if self.released_prefix_span is not None and self.released_prefix_span.start != self.nick_boundary:
                raise ValueError("released_prefix_span.start must equal nick_boundary")
            if self.anchored_duplex_span is not None:
                if self.anchored_duplex_span.start != 0:
                    raise ValueError("anchored_duplex_span.start must be 0 when provided")
                if self.anchored_duplex_span.end != self.nick_boundary:
                    raise ValueError("anchored_duplex_span.end must equal nick_boundary")
            if self.pairings:
                raise ValueError("pre/exposed states must not publish pairings")

        if self.state_kind == "post_nick_foldback":
            if not self.pairings:
                raise ValueError("post_nick_foldback must publish pairings")
            if self.exposed_complement_span is not None:
                raise ValueError("post_nick_foldback must not publish exposed_complement_span")
            if self.released_suffix_span is not None:
                raise ValueError("post_nick_foldback must not publish released_suffix_span")
            if self.loop_geometry is not None:
                if self.cap_span is None:
                    raise ValueError("post_nick_foldback loop_geometry requires cap_span")
                for name, span in (
                    ("loop_geometry.source_cap_span", self.loop_geometry.source_cap_span),
                    ("loop_geometry.cap_extension_span", self.loop_geometry.cap_extension_span),
                    ("loop_geometry.display_primary_span", self.loop_geometry.display_primary_span),
                    ("loop_geometry.display_complement_span", self.loop_geometry.display_complement_span),
                ):
                    if span.end > limit:
                        raise ValueError(f"{name} must lie within primary_sequence bounds")
                if len(range(self.cap_span.start, self.cap_span.end)) != 3:
                    raise ValueError("post_nick_foldback loop_geometry requires cap_span length == 3")
                if self.loop_geometry.display_primary_span != self.retained_stem_span:
                    raise ValueError("loop_geometry.display_primary_span must equal retained_stem_span")
                if self.loop_geometry.display_complement_span != self.foldback_revcomp_span:
                    raise ValueError("loop_geometry.display_complement_span must equal foldback_revcomp_span")
                if len(range(self.retained_stem_span.start, self.retained_stem_span.end)) != len(
                    range(self.foldback_revcomp_span.start, self.foldback_revcomp_span.end)
                ):
                    raise ValueError("loop_geometry display spans must have equal length")
                if self.loop_geometry.source_cap_span.start != self.cap_span.start:
                    raise ValueError("loop_geometry.source_cap_span.start must equal cap_span.start")
                if self.loop_geometry.cap_extension_span.end != self.cap_span.end:
                    raise ValueError("loop_geometry.cap_extension_span.end must equal cap_span.end")
                if self.retained_stem_span.end != self.cap_span.start:
                    raise ValueError("loop_geometry requires retained_stem_span.end == cap_span.start")
                if self.cap_span.end != self.foldback_revcomp_span.start:
                    raise ValueError("loop_geometry requires cap_span.end == foldback_revcomp_span.start")
        elif self.loop_geometry is not None:
            raise ValueError("loop_geometry is only supported for post_nick_foldback")

        if self.released_prefix_span is not None and self.released_prefix_span.end > self.retained_stem_span.start:
            raise ValueError("released_prefix_span must end at or before retained_stem_span.start")
        if self.released_suffix_span is not None and self.released_suffix_span.start < self.retained_stem_span.end:
            raise ValueError("released_suffix_span must start at or after retained_stem_span.end")

        cap_start = self.cap_span.start if self.cap_span is not None else self.foldback_revcomp_span.start
        cap_end = self.cap_span.end if self.cap_span is not None else self.foldback_revcomp_span.start
        if self.retained_stem_span.end > cap_start:
            raise ValueError("retained_stem_span must end at or before cap_span.start")
        if cap_end > self.foldback_revcomp_span.start:
            raise ValueError("cap_span must end at or before foldback_revcomp_span.start")

        if self.released_suffix_span is not None and self.released_suffix_span.end > cap_start:
            raise ValueError("released_suffix_span must end at or before cap_span.start")

        for index in self.primary_mismatch_positions:
            if index < 0 or index >= limit:
                raise ValueError("primary_mismatch_positions must lie within primary_sequence bounds")
        for index in self.complement_mismatch_positions:
            if index < 0 or index >= limit:
                raise ValueError("complement_mismatch_positions must lie within complement_sequence bounds")

        for pair in self.pairings:
            if pair.left_index < self.retained_stem_span.start or pair.left_index >= self.retained_stem_span.end:
                raise ValueError("pairings left_index must remain inside retained_stem_span")
            if (
                pair.right_index < self.foldback_revcomp_span.start
                or pair.right_index >= self.foldback_revcomp_span.end
            ):
                raise ValueError("pairings right_index must remain inside foldback_revcomp_span")

        return self
