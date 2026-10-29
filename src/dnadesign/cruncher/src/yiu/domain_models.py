"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/yiu/domain_models.py

Pure-domain models for normalized YIU payloads and optimization outcomes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.spec_pwm_models import YiuPwmMotifInstanceV1

_LIGATION_POSITIONS = {0, 3}


class JunctionSelection(StrictBaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    offsets: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    mode: Literal["center_locked", "explicit_window", "optimize"]
    left_body_length: int = Field(ge=1)
    right_body_length: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "JunctionSelection":
        if self.end - self.start != 4:
            raise ValueError("junction width must equal 4")
        if self.offsets != [0, 1, 2, 3]:
            raise ValueError("junction offsets must equal [0, 1, 2, 3]")
        return self


class MismatchSelection(StrictBaseModel):
    payload_index: int = Field(ge=0)
    junction_offset: int = Field(ge=0, le=3)
    mutated_strand: Literal["payload", "complement"]
    native_base: str
    mutated_base: str
    opposing_base: str

    @model_validator(mode="after")
    def _validate_bases(self) -> "MismatchSelection":
        for field_name in ("native_base", "mutated_base", "opposing_base"):
            value = getattr(self, field_name)
            if value not in {"A", "C", "G", "T"}:
                raise ValueError(f"{field_name} must be one of A/C/G/T")
        if self.native_base == self.mutated_base:
            raise ValueError("mutated_base must differ from native_base")
        return self


class LigationMismatchRationale(StrictBaseModel):
    payload_index: int = Field(ge=0)
    junction_offset: int = Field(ge=0, le=3)
    position_class: Literal["edge", "middle"]
    mutated_strand: Literal["payload", "complement"]
    native_base: str
    partner_base: str
    canonical_mismatch_class: str
    class_tier: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_values(self) -> "LigationMismatchRationale":
        for field_name in ("native_base", "partner_base"):
            value = getattr(self, field_name)
            if value not in {"A", "C", "G", "T"}:
                raise ValueError(f"{field_name} must be one of A/C/G/T")
        mismatch_class = self.canonical_mismatch_class
        if len(mismatch_class) not in {2}:
            raise ValueError("canonical_mismatch_class must contain two bases")
        if any(base not in {"A", "C", "G", "T"} for base in mismatch_class):
            raise ValueError("canonical_mismatch_class must contain A/C/G/T only")
        return self


class ChosenLigationKey(StrictBaseModel):
    worst_mismatch_class_tier: int = Field(ge=0)
    total_mismatch_class_tier: int = Field(ge=0)
    middle_mismatch_count: int = Field(ge=0)
    double_middle_flag: bool = False
    bad_pattern_penalty: int = Field(ge=0)


class LigationSearchState(StrictBaseModel):
    profile: Literal["none", "t4", "t7", "t3", "pbcv1", "hlig3"] = "none"
    awareness_mode: Literal["disabled", "secondary"] = "disabled"
    selection_mode: Literal["secondary", "pwm_tolerance_then_ligation", "hard_ligation_filter"] = "secondary"
    enabled: bool = False
    legacy_mode: bool = True
    candidate_positions: list[int] = Field(default_factory=list)
    edge_positions_available: bool = False
    edge_comparison_available: bool = False
    pwm_worst_loss_tolerance: float = Field(default=0.0, ge=0.0)
    pwm_total_loss_tolerance: float = Field(default=0.0, ge=0.0)
    max_worst_mismatch_class_tier: int = Field(default=0, ge=0, le=3)
    max_middle_mismatch_count: int = Field(default=1, ge=0, le=2)
    allow_double_middle: bool = False
    allow_tnna_like_overhangs: bool = False
    state_note: str = "Legacy geometric ranking applied"

    @model_validator(mode="after")
    def _validate_state(self) -> "LigationSearchState":
        if self.legacy_mode and self.profile != "none":
            raise ValueError("legacy ligation mode requires ligation_profile=none")
        if self.enabled and self.legacy_mode:
            raise ValueError("enabled ligation ranking cannot be legacy mode")
        if self.edge_comparison_available and not self.enabled:
            raise ValueError("edge comparison cannot be available when ligation ranking is disabled")
        if self.edge_comparison_available and not self.edge_positions_available:
            raise ValueError("edge comparison requires candidate_positions to include 0 or 3")
        return self


class LigationPolicyDecision(StrictBaseModel):
    selection_mode: Literal["secondary", "pwm_tolerance_then_ligation", "hard_ligation_filter"] = "secondary"
    filter_applied: bool = False
    candidate_count_before_filter: int = Field(ge=1)
    candidate_count_after_filter: int = Field(ge=1)
    filtered_candidate_count: int = Field(ge=0)
    fallback_outcome: str | None = None

    @model_validator(mode="after")
    def _validate_counts(self) -> "LigationPolicyDecision":
        if self.candidate_count_after_filter > self.candidate_count_before_filter:
            raise ValueError("candidate_count_after_filter cannot exceed candidate_count_before_filter")
        if self.filtered_candidate_count != self.candidate_count_before_filter - self.candidate_count_after_filter:
            raise ValueError("filtered_candidate_count must equal before_filter - after_filter")
        return self


class OptimizationTraceSample(StrictBaseModel):
    sample_limit: int = Field(ge=0)
    sampled_count: int = Field(ge=0)
    candidate_count: int = Field(ge=1)
    truncated: bool = False

    @model_validator(mode="after")
    def _validate_sample(self) -> "OptimizationTraceSample":
        if self.sampled_count > self.candidate_count:
            raise ValueError("sampled_count cannot exceed candidate_count")
        if self.sampled_count > self.sample_limit:
            raise ValueError("sampled_count cannot exceed sample_limit")
        if self.truncated != (self.candidate_count > self.sample_limit):
            raise ValueError("truncated must reflect whether the trace exceeded sample_limit")
        return self


class NormalizedMotifContext(StrictBaseModel):
    requested_mode: Literal["none", "use_if_available", "require"]
    effective: bool
    source_kind: Literal["none", "sample_context", "file", "inline"]
    fallback_reason: str | None = None
    motifs: list[YiuPwmMotifInstanceV1] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_state(self) -> "NormalizedMotifContext":
        if self.effective and not self.motifs:
            raise ValueError("effective motif_context must contain motifs")
        if not self.effective and self.motifs:
            raise ValueError("ineffective motif_context must not carry resolved motifs")
        return self


class OptimizationObjective(StrictBaseModel):
    primary: Literal["maximin"] = "maximin"
    secondary: list[str]


class OptimizationWinner(StrictBaseModel):
    junction_start: int = Field(ge=0)
    junction_end: int = Field(ge=1)
    selected_positions: list[int] = Field(default_factory=list)
    mutated_strands: list[Literal["payload", "complement"]] = Field(default_factory=list)
    mutated_bases: list[str] = Field(default_factory=list)
    worst_loss: float = Field(ge=0.0)
    total_loss: float = Field(ge=0.0)
    midpoint_distance: int = Field(ge=0)
    middle_mismatch_count: int = Field(ge=0)
    double_middle_flag: bool = False
    default_strand_preference_count: int = Field(ge=0)
    lexical_key: str

    @model_validator(mode="after")
    def _validate_lengths(self) -> "OptimizationWinner":
        if self.junction_end - self.junction_start != 4:
            raise ValueError("winner junction window must be length 4")
        if len(self.selected_positions) != len(self.mutated_strands):
            raise ValueError("selected_positions and mutated_strands must align")
        if len(self.selected_positions) != len(self.mutated_bases):
            raise ValueError("selected_positions and mutated_bases must align")
        return self


class OptimizationDecision(StrictBaseModel):
    candidate_count: int = Field(ge=1)
    objective: OptimizationObjective
    winner: OptimizationWinner
    ligation_policy: LigationPolicyDecision | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    trace_sample: OptimizationTraceSample | None = None


class NormalizedPayload(StrictBaseModel):
    contract: Literal["yiu_normalized_payload_v5"] = "yiu_normalized_payload_v5"
    schema_version: Literal[1] = 1
    name: str
    input_kind: Literal["user_sequence", "sample_hit"]
    payload_label: str | None = None
    site_label: str | None = None
    reference_payload_sequence: str
    reference_complement_sequence: str
    selected_payload_sequence: str
    selected_complement_sequence: str
    source_provenance: dict[str, Any] = Field(default_factory=dict)
    ligation_profile: Literal["none", "t4", "t7", "t3", "pbcv1", "hlig3"] = "none"
    ligation_awareness_mode: Literal["disabled", "secondary"] = "disabled"
    ligation_selection_mode: Literal["secondary", "pwm_tolerance_then_ligation", "hard_ligation_filter"] = "secondary"
    bad_pattern_heuristics: bool = False
    ligation_state: LigationSearchState | None = None
    chosen_ligation_key: ChosenLigationKey | None = None
    ligation_rationale: list[LigationMismatchRationale] = Field(default_factory=list)
    junction: JunctionSelection
    mismatches: list[MismatchSelection] = Field(default_factory=list)
    motif_context: NormalizedMotifContext
    optimization_decision: OptimizationDecision

    @model_validator(mode="after")
    def _validate_payload(self) -> "NormalizedPayload":
        payload_length = len(self.reference_payload_sequence)
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        for field_name in (
            "reference_complement_sequence",
            "selected_payload_sequence",
            "selected_complement_sequence",
        ):
            if len(getattr(self, field_name)) != payload_length:
                raise ValueError(f"{field_name} length must match reference_payload_sequence")
        if self.junction.end > payload_length:
            raise ValueError("junction end must lie within the payload")
        expected_positions = {self.junction.start + offset for offset in range(4)}
        seen_positions: set[int] = set()
        for mismatch in self.mismatches:
            if mismatch.payload_index not in expected_positions:
                raise ValueError("mismatch payload_index must lie inside the selected junction window")
            if mismatch.payload_index in seen_positions:
                raise ValueError("mismatch payload_index values must be unique")
            seen_positions.add(mismatch.payload_index)
        if self.chosen_ligation_key is None and self.ligation_rationale:
            raise ValueError("ligation_rationale requires chosen_ligation_key")
        if self.ligation_state is not None:
            if self.ligation_state.profile != self.ligation_profile:
                raise ValueError("ligation_state.profile must match ligation_profile")
            if self.ligation_state.awareness_mode != self.ligation_awareness_mode:
                raise ValueError("ligation_state.awareness_mode must match ligation_awareness_mode")
            if self.ligation_state.selection_mode != self.ligation_selection_mode:
                raise ValueError("ligation_state.selection_mode must match ligation_selection_mode")
        if self.optimization_decision.trace_sample is not None:
            if self.optimization_decision.trace_sample.candidate_count != self.optimization_decision.candidate_count:
                raise ValueError("trace_sample.candidate_count must match candidate_count")
            if self.optimization_decision.trace_sample.sampled_count != len(self.optimization_decision.trace):
                raise ValueError("trace_sample.sampled_count must match sampled trace length")
        if self.optimization_decision.ligation_policy is not None:
            if (
                self.optimization_decision.ligation_policy.candidate_count_before_filter
                != self.optimization_decision.candidate_count
            ):
                raise ValueError("ligation_policy.candidate_count_before_filter must match candidate_count")
        return self

    @property
    def payload_length(self) -> int:
        return len(self.reference_payload_sequence)


def build_ligation_search_state(
    *,
    ligation_profile: Literal["none", "t4", "t7", "t3", "pbcv1", "hlig3"],
    ligation_awareness_mode: Literal["disabled", "secondary"],
    ligation_selection_mode: Literal["secondary", "pwm_tolerance_then_ligation", "hard_ligation_filter"] = "secondary",
    candidate_positions: list[int] | tuple[int, ...],
    pwm_worst_loss_tolerance: float = 0.0,
    pwm_total_loss_tolerance: float = 0.0,
    max_worst_mismatch_class_tier: int = 0,
    max_middle_mismatch_count: int = 1,
    allow_double_middle: bool = False,
    allow_tnna_like_overhangs: bool = False,
) -> LigationSearchState:
    positions = [int(position) for position in candidate_positions]
    enabled = ligation_awareness_mode == "secondary" and ligation_profile != "none"
    legacy_mode = ligation_profile == "none"
    edge_positions_available = any(position in _LIGATION_POSITIONS for position in positions)
    if legacy_mode:
        state_note = "Legacy mode because ligation_profile=none"
    elif not enabled:
        state_note = "Ligation-aware scoring disabled by config"
    elif ligation_selection_mode == "hard_ligation_filter":
        state_note = (
            "Hard ligation filter active; edge-vs-middle comparison unavailable because "
            "candidate_positions excludes 0/3"
            if not edge_positions_available
            else "Hard ligation filter active before PWM-preserving ranking"
        )
    elif ligation_selection_mode == "pwm_tolerance_then_ligation":
        state_note = (
            "PWM tolerance gate active; edge-vs-middle comparison unavailable because candidate_positions excludes 0/3"
            if not edge_positions_available
            else "PWM tolerance gate active before ligation-aware ranking"
        )
    elif not edge_positions_available:
        state_note = (
            "Ligation-aware scoring active; edge-vs-middle comparison unavailable because "
            "candidate_positions excludes 0/3"
        )
    else:
        state_note = "Ligation-aware scoring active; edge-vs-middle comparison available"
    return LigationSearchState(
        profile=ligation_profile,
        awareness_mode=ligation_awareness_mode,
        selection_mode=ligation_selection_mode,
        enabled=enabled,
        legacy_mode=legacy_mode,
        candidate_positions=sorted(positions),
        edge_positions_available=edge_positions_available,
        edge_comparison_available=enabled and edge_positions_available,
        pwm_worst_loss_tolerance=pwm_worst_loss_tolerance,
        pwm_total_loss_tolerance=pwm_total_loss_tolerance,
        max_worst_mismatch_class_tier=max_worst_mismatch_class_tier,
        max_middle_mismatch_count=max_middle_mismatch_count,
        allow_double_middle=allow_double_middle,
        allow_tnna_like_overhangs=allow_tnna_like_overhangs,
        state_note=state_note,
    )


def build_trace_sample(
    *,
    candidate_count: int,
    sample_limit: int,
    sampled_count: int,
) -> OptimizationTraceSample:
    return OptimizationTraceSample(
        candidate_count=candidate_count,
        sample_limit=sample_limit,
        sampled_count=sampled_count,
        truncated=candidate_count > sample_limit,
    )
