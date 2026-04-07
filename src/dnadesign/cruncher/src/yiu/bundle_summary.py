"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/bundle_summary.py

Operator-facing summary models and builders for YIU bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.bsmbi import SplitFragmentDisplaySpec, build_split_fragment_display_specs
from dnadesign.cruncher.yiu.bundle_models import (
    PayloadVisualInventory,
    resolve_ligation_decision_note,
    resolve_ligation_surface_state,
)
from dnadesign.cruncher.yiu.bundle_models import (
    build_trace_summary as build_validation_trace_summary,
)
from dnadesign.cruncher.yiu.domain_models import (
    ChosenLigationKey,
    JunctionSelection,
    LigationMismatchRationale,
    MismatchSelection,
    NormalizedPayload,
)
from dnadesign.cruncher.yiu.mismatch_notation import compact_mismatch_notation_groups


class YiuStrandPair5to3Summary(StrictBaseModel):
    top_strand_5to3: str
    bottom_strand_5to3: str


class YiuViewSequenceSummary(StrictBaseModel):
    canonical: YiuStrandPair5to3Summary
    mismatch_present: YiuStrandPair5to3Summary
    changed_rows: list[Literal["top", "bottom"]] = Field(default_factory=list)


class YiuSequenceViewsSummary(StrictBaseModel):
    payload: YiuViewSequenceSummary
    split_left: YiuViewSequenceSummary
    split_right: YiuViewSequenceSummary
    assembled: YiuViewSequenceSummary


class YiuOverhangSummary(StrictBaseModel):
    canonical_sequence_5to3: str
    mismatch_present_sequence_5to3: str


class YiuSequenceSummary(StrictBaseModel):
    junction_payload_sequence_5to3: str
    overhang_5to3: YiuOverhangSummary
    views: YiuSequenceViewsSummary


class YiuPwmSummary(StrictBaseModel):
    mode: Literal["none", "use_if_available", "require"]
    effective: bool
    motif_count: int = Field(ge=0)
    worst_loss: float = Field(ge=0.0)
    total_loss: float = Field(ge=0.0)


class YiuLigationSummary(StrictBaseModel):
    profile: Literal["none", "t4", "t7", "t3", "pbcv1", "hlig3"]
    awareness_mode: Literal["disabled", "secondary"]
    applied: bool
    state: Literal["legacy", "inert", "edge_blind", "active"] | None = None
    state_note: str | None = None
    enabled: bool | None = None
    legacy_mode: bool | None = None
    candidate_positions: list[int] = Field(default_factory=list)
    edge_positions_available: bool | None = None
    edge_comparison_available: bool | None = None
    bad_pattern_heuristics: bool
    chosen_mismatch_classes: list[str] = Field(default_factory=list)
    position_classes: list[Literal["edge", "middle"]] = Field(default_factory=list)
    worst_mismatch_class_tier: int | None = Field(default=None, ge=0)
    total_mismatch_class_tier: int | None = Field(default=None, ge=0)
    middle_mismatch_count: int | None = Field(default=None, ge=0)
    double_middle_flag: bool | None = None
    bad_pattern_penalty: int | None = Field(default=None, ge=0)
    decision_note: str


class YiuTraceSummary(StrictBaseModel):
    sample_limit: int = Field(ge=0)
    sampled_count: int = Field(ge=0)
    candidate_count: int = Field(ge=1)
    truncated: bool
    note: str

    @property
    def sample_count(self) -> int:
        return self.sampled_count


class YiuBundleSummary(StrictBaseModel):
    summary_contract: Literal["yiu_bundle_summary_v3"] = "yiu_bundle_summary_v3"
    schema_version: Literal[3] = 3
    spec_name: str
    payload_label: str | None = None
    input_kind: Literal["user_sequence", "sample_hit"]
    payload_length: int = Field(ge=1)
    sequence_summary: YiuSequenceSummary
    junction: JunctionSelection
    mismatches: list[MismatchSelection] = Field(default_factory=list)
    mismatch_notation: list[str] = Field(default_factory=list)
    pwm: YiuPwmSummary
    ligation: YiuLigationSummary
    trace: YiuTraceSummary | None = None
    view_ids: list[str] = Field(default_factory=list)
    render_status: Literal["not_requested", "rendered", "missing", "partial", "failed"] = "not_requested"


def _strand_pair(*, top_strand_5to3: str, bottom_strand_5to3: str) -> YiuStrandPair5to3Summary:
    return YiuStrandPair5to3Summary(
        top_strand_5to3=top_strand_5to3,
        bottom_strand_5to3=bottom_strand_5to3,
    )


def _changed_rows(
    canonical: YiuStrandPair5to3Summary,
    mismatch_present: YiuStrandPair5to3Summary,
) -> list[Literal["top", "bottom"]]:
    rows: list[Literal["top", "bottom"]] = []
    if canonical.top_strand_5to3 != mismatch_present.top_strand_5to3:
        rows.append("top")
    if canonical.bottom_strand_5to3 != mismatch_present.bottom_strand_5to3:
        rows.append("bottom")
    return rows


def _view_summary(
    *,
    canonical: YiuStrandPair5to3Summary,
    mismatch_present: YiuStrandPair5to3Summary,
) -> YiuViewSequenceSummary:
    return YiuViewSequenceSummary(
        canonical=canonical,
        mismatch_present=mismatch_present,
        changed_rows=_changed_rows(canonical, mismatch_present),
    )


def _payload_view_strands(
    *,
    payload_sequence_5to3: str,
    complement_sequence_3to5: str,
) -> YiuStrandPair5to3Summary:
    return _strand_pair(
        top_strand_5to3=payload_sequence_5to3,
        bottom_strand_5to3=complement_sequence_3to5[::-1],
    )


def _split_fragment_strands(fragment: SplitFragmentDisplaySpec) -> YiuStrandPair5to3Summary:
    return _strand_pair(
        top_strand_5to3=fragment.retained_primary_sequence_5to3,
        bottom_strand_5to3=fragment.retained_complement_sequence_3to5[::-1],
    )


def _canonical_normalized(normalized: NormalizedPayload) -> NormalizedPayload:
    return normalized.model_copy(
        update={
            "selected_payload_sequence": normalized.reference_payload_sequence,
            "selected_complement_sequence": normalized.reference_complement_sequence,
        }
    )


def _trace_summary(normalized: NormalizedPayload) -> YiuTraceSummary:
    return YiuTraceSummary.model_validate(
        build_validation_trace_summary(
            candidate_count=normalized.optimization_decision.candidate_count,
            trace_sample=normalized.optimization_decision.trace_sample,
            trace_len=len(normalized.optimization_decision.trace),
        ).model_dump(mode="json")
    )


def build_sequence_summary(normalized: NormalizedPayload) -> YiuSequenceSummary:
    junction_start = normalized.junction.start
    junction_end = normalized.junction.end
    canonical_normalized = _canonical_normalized(normalized)
    canonical_split_fragments = {
        fragment.fragment_side: fragment for fragment in build_split_fragment_display_specs(canonical_normalized)
    }
    selected_split_fragments = {
        fragment.fragment_side: fragment for fragment in build_split_fragment_display_specs(normalized)
    }
    canonical_payload = _payload_view_strands(
        payload_sequence_5to3=normalized.reference_payload_sequence,
        complement_sequence_3to5=normalized.reference_complement_sequence,
    )
    mismatch_present_payload = _payload_view_strands(
        payload_sequence_5to3=normalized.selected_payload_sequence,
        complement_sequence_3to5=normalized.selected_complement_sequence,
    )
    return YiuSequenceSummary(
        junction_payload_sequence_5to3=normalized.selected_payload_sequence[junction_start:junction_end],
        overhang_5to3=YiuOverhangSummary(
            canonical_sequence_5to3=normalized.reference_complement_sequence[junction_start:junction_end][::-1],
            mismatch_present_sequence_5to3=normalized.selected_complement_sequence[junction_start:junction_end][::-1],
        ),
        views=YiuSequenceViewsSummary(
            payload=_view_summary(
                canonical=canonical_payload,
                mismatch_present=mismatch_present_payload,
            ),
            split_left=_view_summary(
                canonical=_split_fragment_strands(canonical_split_fragments["left"]),
                mismatch_present=_split_fragment_strands(selected_split_fragments["left"]),
            ),
            split_right=_view_summary(
                canonical=_split_fragment_strands(canonical_split_fragments["right"]),
                mismatch_present=_split_fragment_strands(selected_split_fragments["right"]),
            ),
            assembled=_view_summary(
                canonical=canonical_payload,
                mismatch_present=mismatch_present_payload,
            ),
        ),
    )


def build_bundle_summary(
    *,
    normalized: NormalizedPayload,
    inventory: PayloadVisualInventory,
) -> YiuBundleSummary:
    ligation_key: ChosenLigationKey | None = normalized.chosen_ligation_key
    ligation_rationale: list[LigationMismatchRationale] = normalized.ligation_rationale
    ligation_applied = ligation_key is not None
    ligation_state, ligation_state_note, edge_comparison_available = resolve_ligation_surface_state(normalized)
    trace_summary = _trace_summary(normalized)
    return YiuBundleSummary(
        spec_name=inventory.spec_name,
        payload_label=normalized.payload_label,
        input_kind=normalized.input_kind,
        payload_length=normalized.payload_length,
        sequence_summary=build_sequence_summary(normalized),
        junction=normalized.junction,
        mismatches=normalized.mismatches,
        mismatch_notation=compact_mismatch_notation_groups(normalized.mismatches),
        pwm=YiuPwmSummary(
            mode=normalized.motif_context.requested_mode,
            effective=normalized.motif_context.effective,
            motif_count=len(normalized.motif_context.motifs),
            worst_loss=normalized.optimization_decision.winner.worst_loss,
            total_loss=normalized.optimization_decision.winner.total_loss,
        ),
        ligation=YiuLigationSummary(
            profile=normalized.ligation_profile,
            awareness_mode=normalized.ligation_awareness_mode,
            applied=ligation_applied,
            state=ligation_state,
            state_note=ligation_state_note,
            enabled=False if normalized.ligation_state is None else normalized.ligation_state.enabled,
            legacy_mode=True if normalized.ligation_state is None else normalized.ligation_state.legacy_mode,
            candidate_positions=[]
            if normalized.ligation_state is None
            else list(normalized.ligation_state.candidate_positions),
            edge_positions_available=False
            if normalized.ligation_state is None
            else normalized.ligation_state.edge_positions_available,
            edge_comparison_available=edge_comparison_available,
            bad_pattern_heuristics=normalized.bad_pattern_heuristics,
            chosen_mismatch_classes=[entry.canonical_mismatch_class for entry in ligation_rationale],
            position_classes=[entry.position_class for entry in ligation_rationale],
            worst_mismatch_class_tier=None if ligation_key is None else ligation_key.worst_mismatch_class_tier,
            total_mismatch_class_tier=None if ligation_key is None else ligation_key.total_mismatch_class_tier,
            middle_mismatch_count=None if ligation_key is None else ligation_key.middle_mismatch_count,
            double_middle_flag=None if ligation_key is None else ligation_key.double_middle_flag,
            bad_pattern_penalty=None if ligation_key is None else ligation_key.bad_pattern_penalty,
            decision_note=resolve_ligation_decision_note(
                state=ligation_state,
                ligation_applied=ligation_applied,
                pwm_effective=normalized.motif_context.effective,
            ),
        ),
        trace=trace_summary,
        view_ids=[view.view_id for view in inventory.views],
        render_status=inventory.render_status,
    )


__all__ = [
    "YiuBundleSummary",
    "YiuLigationSummary",
    "YiuOverhangSummary",
    "YiuPwmSummary",
    "YiuSequenceSummary",
    "YiuSequenceViewsSummary",
    "YiuStrandPair5to3Summary",
    "YiuTraceSummary",
    "YiuViewSequenceSummary",
    "build_bundle_summary",
    "build_sequence_summary",
]
