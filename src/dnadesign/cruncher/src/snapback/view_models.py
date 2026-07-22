"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/view_models.py

Producer-owned QA view contracts for snapback visual artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from dnadesign.cruncher.nickases.models import NickEvent, RecognitionSiteInstance, normalize_dna
from dnadesign.cruncher.snapback.models import CoordinateSpan, PairContract, StrictSnapbackModel


def _validate_span_within(span: CoordinateSpan, *, limit: int, label: str) -> None:
    if span.end > limit:
        raise ValueError(f"{label} must stay inside the sequence span.")


class SnapbackStrandRow(StrictSnapbackModel):
    label: str
    direction: Literal["5to3", "3to5"]
    sequence: str

    @model_validator(mode="after")
    def _validate_sequence(self) -> "SnapbackStrandRow":
        normalize_dna(self.sequence)
        return self


class SnapbackDuplexRows(StrictSnapbackModel):
    top: SnapbackStrandRow
    complement: SnapbackStrandRow


class SnapbackExposedTopology(StrictSnapbackModel):
    anchored_top_span: CoordinateSpan
    released_top_span: CoordinateSpan
    released_prefix_span: CoordinateSpan
    retained_homology_span: CoordinateSpan
    source_cap_span: CoordinateSpan
    cap_extension_span: CoordinateSpan
    cap_span: CoordinateSpan
    foldback_arm_span: CoordinateSpan

    @model_validator(mode="after")
    def _validate_order(self) -> "SnapbackExposedTopology":
        if self.anchored_top_span.end > self.released_top_span.start:
            raise ValueError("anchored_top_span must end at or before released_top_span.start.")
        if self.released_prefix_span.start != self.released_top_span.start:
            raise ValueError("released_prefix_span must start at released_top_span.start.")
        if self.released_prefix_span.end > self.retained_homology_span.start:
            raise ValueError("released_prefix_span must end at or before retained_homology_span.start.")
        if self.retained_homology_span.end != self.source_cap_span.start:
            raise ValueError("retained_homology_span must end at source_cap_span.start.")
        if self.source_cap_span.end != self.cap_extension_span.start:
            raise ValueError("source_cap_span must end at cap_extension_span.start.")
        if self.cap_extension_span.end != self.foldback_arm_span.start:
            raise ValueError("cap_extension_span must end at foldback_arm_span.start.")
        if self.cap_span.start != self.source_cap_span.start or self.cap_span.end != self.cap_extension_span.end:
            raise ValueError("cap_span must cover source_cap_span + cap_extension_span.")
        if self.retained_homology_span.end > self.cap_span.start:
            raise ValueError("retained_homology_span must end at or before cap_span.start.")
        if self.released_top_span.end != self.foldback_arm_span.end:
            raise ValueError("released_top_span must end at foldback_arm_span.end.")
        return self


class SnapbackFoldbackTopology(StrictSnapbackModel):
    released_prefix_span: CoordinateSpan
    retained_homology_span: CoordinateSpan
    source_cap_span: CoordinateSpan
    cap_extension_span: CoordinateSpan
    cap_span: CoordinateSpan
    foldback_arm_span: CoordinateSpan
    protected_overlap_span: CoordinateSpan | None = None

    @model_validator(mode="after")
    def _validate_order(self) -> "SnapbackFoldbackTopology":
        if self.released_prefix_span.end > self.retained_homology_span.start:
            raise ValueError("released_prefix_span must end at or before retained_homology_span.start.")
        if self.retained_homology_span.end != self.source_cap_span.start:
            raise ValueError("retained_homology_span must end at source_cap_span.start.")
        if self.source_cap_span.end != self.cap_extension_span.start:
            raise ValueError("source_cap_span must end at cap_extension_span.start.")
        if self.cap_extension_span.end != self.foldback_arm_span.start:
            raise ValueError("cap_extension_span must end at foldback_arm_span.start.")
        if self.cap_span.start != self.source_cap_span.start or self.cap_span.end != self.cap_extension_span.end:
            raise ValueError("cap_span must cover source_cap_span + cap_extension_span.")
        if self.protected_overlap_span is not None:
            if self.protected_overlap_span.start < self.retained_homology_span.start:
                raise ValueError("protected_overlap_span must stay inside retained_homology_span.")
            if self.protected_overlap_span.end > self.retained_homology_span.end:
                raise ValueError("protected_overlap_span must stay inside retained_homology_span.")
        return self


class SnapbackPreNickDuplexViewV1(StrictSnapbackModel):
    version: Literal[1] = 1
    kind: Literal["snapback_pre_nick_duplex_v1"] = "snapback_pre_nick_duplex_v1"
    view_id: str
    solution_id: str
    title: str
    coordinate_semantics: Literal["half_open_zero_based_v1"]
    boundary_semantics: Literal["closed_zero_based_boundary_v1"]
    sequence_span: CoordinateSpan
    input_span: CoordinateSpan
    rows: SnapbackDuplexRows
    nick_boundary: int = Field(ge=0)
    ligation_junction_boundary: int = Field(ge=0)
    protected_region: CoordinateSpan
    pre_nick_duplex_window: CoordinateSpan
    retained_homology_window: CoordinateSpan
    source_cap_window: CoordinateSpan
    effective_cap_window: CoordinateSpan
    cap_span: CoordinateSpan
    foldback_arm_span: CoordinateSpan
    intended_site: RecognitionSiteInstance
    intended_nick: NickEvent
    extra_target_strand_nicks: list[NickEvent] = Field(default_factory=list)
    extra_nick_events: list[NickEvent] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sequence_lengths(self) -> "SnapbackPreNickDuplexViewV1":
        sequence_length = self.sequence_span.end - self.sequence_span.start
        if len(self.rows.top.sequence) != sequence_length:
            raise ValueError("top row length must match sequence_span.")
        if len(self.rows.complement.sequence) != sequence_length:
            raise ValueError("complement row length must match sequence_span.")
        if self.input_span.end > self.sequence_span.end:
            raise ValueError("input_span must stay inside sequence_span.")
        if self.nick_boundary > self.input_span.end:
            raise ValueError("nick_boundary must stay inside input_span.")
        if self.ligation_junction_boundary > self.input_span.end:
            raise ValueError("ligation_junction_boundary must stay inside input_span.")
        if self.ligation_junction_boundary != self.nick_boundary:
            raise ValueError("ligation_junction_boundary must equal nick_boundary in pre-nick duplex view.")
        for label, span in (
            ("protected_region", self.protected_region),
            ("pre_nick_duplex_window", self.pre_nick_duplex_window),
            ("retained_homology_window", self.retained_homology_window),
            ("source_cap_window", self.source_cap_window),
            ("effective_cap_window", self.effective_cap_window),
            ("cap_span", self.cap_span),
            ("foldback_arm_span", self.foldback_arm_span),
        ):
            _validate_span_within(span, limit=sequence_length, label=label)
        if self.effective_cap_window.start != self.source_cap_window.start:
            raise ValueError("effective_cap_window.start must equal source_cap_window.start.")
        if self.effective_cap_window.end != self.cap_span.end:
            raise ValueError("effective_cap_window.end must equal cap_span.end.")
        return self


class SnapbackPostNickExposedViewV1(StrictSnapbackModel):
    version: Literal[1] = 1
    kind: Literal["snapback_post_nick_exposed_v1"] = "snapback_post_nick_exposed_v1"
    view_id: str
    solution_id: str
    title: str
    coordinate_semantics: Literal["half_open_zero_based_v1"]
    boundary_semantics: Literal["closed_zero_based_boundary_v1"]
    sequence_span: CoordinateSpan
    rows: SnapbackDuplexRows
    nick_boundary: int = Field(ge=0)
    ligation_junction_boundary: int = Field(ge=0)
    topology: SnapbackExposedTopology
    intended_site: RecognitionSiteInstance
    intended_nick: NickEvent
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sequence_lengths(self) -> "SnapbackPostNickExposedViewV1":
        sequence_length = self.sequence_span.end - self.sequence_span.start
        if len(self.rows.top.sequence) != sequence_length:
            raise ValueError("top row length must match sequence_span.")
        if len(self.rows.complement.sequence) != sequence_length:
            raise ValueError("complement row length must match sequence_span.")
        if self.nick_boundary > self.sequence_span.end:
            raise ValueError("nick_boundary must stay inside sequence_span.")
        if self.ligation_junction_boundary > self.sequence_span.end:
            raise ValueError("ligation_junction_boundary must stay inside sequence_span.")
        if self.ligation_junction_boundary != self.nick_boundary:
            raise ValueError("ligation_junction_boundary must equal nick_boundary in post-nick exposed view.")
        if self.topology.anchored_top_span.end != self.nick_boundary:
            raise ValueError("anchored_top_span.end must equal nick_boundary.")
        if self.topology.retained_homology_span.start != self.ligation_junction_boundary:
            raise ValueError("retained_homology_span.start must equal ligation_junction_boundary.")
        for label, span in (
            ("anchored_top_span", self.topology.anchored_top_span),
            ("released_top_span", self.topology.released_top_span),
            ("released_prefix_span", self.topology.released_prefix_span),
            ("retained_homology_span", self.topology.retained_homology_span),
            ("source_cap_span", self.topology.source_cap_span),
            ("cap_extension_span", self.topology.cap_extension_span),
            ("cap_span", self.topology.cap_span),
            ("foldback_arm_span", self.topology.foldback_arm_span),
        ):
            _validate_span_within(span, limit=sequence_length, label=f"topology.{label}")
        return self


class SnapbackPostNickFoldbackViewV1(StrictSnapbackModel):
    version: Literal[1] = 1
    kind: Literal["snapback_post_nick_foldback_v1"] = "snapback_post_nick_foldback_v1"
    view_id: str
    solution_id: str
    title: str
    coordinate_semantics: Literal["half_open_zero_based_v1"]
    boundary_semantics: Literal["closed_zero_based_boundary_v1"]
    source_nick_boundary: int = Field(ge=0)
    ligation_junction_boundary: int = Field(ge=0)
    primary_sequence_5to3: str
    topology: SnapbackFoldbackTopology
    pair_map: list[PairContract] = Field(default_factory=list)
    primary_mismatch_positions: list[int] = Field(default_factory=list)
    foldback_partner_mismatch_positions: list[int] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sequence(self) -> "SnapbackPostNickFoldbackViewV1":
        normalize_dna(self.primary_sequence_5to3, allow_empty=False)
        sequence_length = len(self.primary_sequence_5to3)
        if self.ligation_junction_boundary != self.topology.retained_homology_span.start:
            raise ValueError("ligation_junction_boundary must equal topology.retained_homology_span.start.")
        for label, span in (
            ("topology.released_prefix_span", self.topology.released_prefix_span),
            ("topology.retained_homology_span", self.topology.retained_homology_span),
            ("topology.source_cap_span", self.topology.source_cap_span),
            ("topology.cap_extension_span", self.topology.cap_extension_span),
            ("topology.cap_span", self.topology.cap_span),
            ("topology.foldback_arm_span", self.topology.foldback_arm_span),
        ):
            _validate_span_within(span, limit=sequence_length, label=label)
        if self.topology.protected_overlap_span is not None:
            _validate_span_within(
                self.topology.protected_overlap_span,
                limit=sequence_length,
                label="topology.protected_overlap_span",
            )
        if any(
            position < self.topology.retained_homology_span.start
            or position >= self.topology.retained_homology_span.end
            for position in self.primary_mismatch_positions
        ):
            raise ValueError("primary_mismatch_positions must stay inside topology.retained_homology_span.")
        if any(
            position < self.topology.foldback_arm_span.start or position >= self.topology.foldback_arm_span.end
            for position in self.foldback_partner_mismatch_positions
        ):
            raise ValueError("foldback_partner_mismatch_positions must stay inside topology.foldback_arm_span.")
        for pair in self.pair_map:
            if not (self.topology.retained_homology_span.start <= pair.left < self.topology.retained_homology_span.end):
                raise ValueError("pair_map left indices must stay inside topology.retained_homology_span.")
            if not (self.topology.foldback_arm_span.start <= pair.right < self.topology.foldback_arm_span.end):
                raise ValueError("pair_map right indices must stay inside topology.foldback_arm_span.")
        return self


class SnapbackViewsManifestEntryV1(StrictSnapbackModel):
    name: str
    path: str
    contract_kind: str


class SnapbackRecommendedJobEntryV1(StrictSnapbackModel):
    name: str
    path: str


class SnapbackViewsManifestV1(StrictSnapbackModel):
    version: Literal[1] = 1
    kind: Literal["snapback_views_manifest_v1"] = "snapback_views_manifest_v1"
    solution_id: str
    views: list[SnapbackViewsManifestEntryV1]
    recommended_jobs: list[SnapbackRecommendedJobEntryV1] = Field(default_factory=list)
