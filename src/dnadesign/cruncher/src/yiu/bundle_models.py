"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/bundle_models.py

Bundle, inventory, and summary models for YIU v4 publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.domain_models import (
    JunctionSelection,
    LigationSearchState,
    MismatchSelection,
    NormalizedPayload,
    OptimizationTraceSample,
)

_SUMMARY_FIELDS = (
    "payload_label",
    "input_kind",
    "payload_length",
    "selected_payload_sequence",
    "selected_complement_sequence",
    "junction",
    "mismatches",
    "pwm_mode",
    "pwm_effective",
    "worst_loss",
    "total_loss",
)


class YiuValidationIssue(StrictBaseModel):
    code: str
    message: str


class YiuValidationLigationSummary(StrictBaseModel):
    profile: Literal["none", "t4", "t7", "t3", "pbcv1", "hlig3"]
    awareness_mode: Literal["disabled", "secondary"]
    selection_mode: Literal["secondary", "pwm_tolerance_then_ligation", "hard_ligation_filter"] = "secondary"
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
    candidate_count_before_filter: int | None = Field(default=None, ge=1)
    candidate_count_after_filter: int | None = Field(default=None, ge=1)
    filtered_candidate_count: int | None = Field(default=None, ge=0)
    decision_note: str


class YiuValidationTraceSummary(StrictBaseModel):
    sample_limit: int = Field(ge=0)
    sampled_count: int = Field(ge=0)
    candidate_count: int = Field(ge=1)
    truncated: bool
    note: str


class YiuValidationReport(StrictBaseModel):
    workflow: Literal["yiu"] = "yiu"
    contract: Literal["split_yiu_payload_rendering_v4"] = "split_yiu_payload_rendering_v4"
    spec_name: str
    status: Literal["satisfied", "unsatisfied"]
    input_kind: Literal["user_sequence", "sample_hit"]
    payload_label: str | None = None
    payload_length: int = Field(ge=1)
    selected_payload_sequence: str
    selected_complement_sequence: str
    junction: JunctionSelection
    mismatches: list[MismatchSelection] = Field(default_factory=list)
    pwm_mode: Literal["none", "use_if_available", "require"]
    pwm_effective: bool
    worst_loss: float = Field(ge=0.0)
    total_loss: float = Field(ge=0.0)
    ligation: YiuValidationLigationSummary
    trace: YiuValidationTraceSummary | None = None
    bundle_dir: str | None = None
    issues: list[YiuValidationIssue] = Field(default_factory=list)


class PayloadViewEntry(StrictBaseModel):
    view_id: Literal["payload", "split_payload", "assembled_payload"]
    visual_direction: str
    contract_kind: str
    schema_version: int = 1
    input_kind: Literal["json", "jsonl"]
    view_contract_path: str
    render_artifact_path: str
    renderer_kind: str
    style_preset: str | None = None
    style_overrides: dict[str, object] = Field(default_factory=dict)
    render_requested: bool = False
    render_completed: bool = False
    last_rendered_at: str | None = None
    motif_layers_required: bool = False


class PayloadBundleManifest(StrictBaseModel):
    bundle_contract: Literal["split_yiu_payload_bundle_v4"] = "split_yiu_payload_bundle_v4"
    input_contract: Literal["split_yiu_payload_rendering_v4"] = "split_yiu_payload_rendering_v4"
    spec_name: str
    input_kind: Literal["user_sequence", "sample_hit"]
    payload_label: str | None = None
    payload_length: int = Field(ge=1)
    selected_payload_sequence: str
    selected_complement_sequence: str
    junction: JunctionSelection
    mismatches: list[MismatchSelection] = Field(default_factory=list)
    pwm_mode: Literal["none", "use_if_available", "require"]
    pwm_effective: bool
    worst_loss: float = Field(ge=0.0)
    total_loss: float = Field(ge=0.0)
    provenance: dict[str, object] = Field(default_factory=dict)
    payload_view_requires_motif_layers: bool = False
    view_contracts: list[PayloadViewEntry] = Field(default_factory=list)
    composite_render_artifact_path: str | None = None
    published_plot_artifact_path: str | None = None
    render_status: Literal["not_requested", "rendered", "missing", "partial", "failed"] = "not_requested"


class PayloadVisualInventory(StrictBaseModel):
    bundle_contract: Literal["split_yiu_payload_bundle_v4"] = "split_yiu_payload_bundle_v4"
    input_contract: Literal["split_yiu_payload_rendering_v4"] = "split_yiu_payload_rendering_v4"
    spec_name: str
    input_kind: Literal["user_sequence", "sample_hit"]
    view_count: int = Field(ge=0)
    render_count: int = Field(ge=0)
    render_status: Literal["not_requested", "rendered", "missing", "partial", "failed"] = "not_requested"
    last_rendered_at: str | None = None
    composite_render_artifact_path: str | None = None
    published_plot_artifact_path: str | None = None
    pwm_effective: bool = False
    payload_view_requires_motif_layers: bool = False
    views: list[PayloadViewEntry] = Field(default_factory=list)


def payload_summary_from_normalized(normalized: NormalizedPayload) -> dict[str, object]:
    winner = normalized.optimization_decision.winner
    return {
        "input_kind": normalized.input_kind,
        "payload_label": normalized.payload_label,
        "payload_length": normalized.payload_length,
        "selected_payload_sequence": normalized.selected_payload_sequence,
        "selected_complement_sequence": normalized.selected_complement_sequence,
        "junction": normalized.junction,
        "mismatches": normalized.mismatches,
        "pwm_mode": normalized.motif_context.requested_mode,
        "pwm_effective": normalized.motif_context.effective,
        "worst_loss": winner.worst_loss,
        "total_loss": winner.total_loss,
    }


def normalized_payload_summary_dump(normalized: NormalizedPayload) -> dict[str, object]:
    return {
        "payload_label": normalized.payload_label,
        "input_kind": normalized.input_kind,
        "payload_length": normalized.payload_length,
        "selected_payload_sequence": normalized.selected_payload_sequence,
        "selected_complement_sequence": normalized.selected_complement_sequence,
        "junction": normalized.junction.model_dump(mode="json"),
        "mismatches": [entry.model_dump(mode="json") for entry in normalized.mismatches],
        "pwm_mode": normalized.motif_context.requested_mode,
        "pwm_effective": normalized.motif_context.effective,
        "worst_loss": normalized.optimization_decision.winner.worst_loss,
        "total_loss": normalized.optimization_decision.winner.total_loss,
    }


def payload_summary_dump(summary: YiuValidationReport | PayloadBundleManifest) -> dict[str, object]:
    return summary.model_dump(mode="json", include=set(_SUMMARY_FIELDS))


def resolve_ligation_surface_state(
    normalized: NormalizedPayload,
) -> tuple[Literal["legacy", "inert", "edge_blind", "active"], str, bool]:
    ligation_state: LigationSearchState | None = normalized.ligation_state
    if ligation_state is None:
        return "inert", "Ligation state is unavailable in the normalized payload.", False
    if ligation_state.legacy_mode:
        return "legacy", ligation_state.state_note, ligation_state.edge_comparison_available
    if not ligation_state.enabled:
        return "inert", ligation_state.state_note, ligation_state.edge_comparison_available
    if not ligation_state.edge_comparison_available:
        return "edge_blind", ligation_state.state_note, False
    return "active", ligation_state.state_note, True


def resolve_ligation_decision_note(
    *,
    state: Literal["legacy", "inert", "edge_blind", "active"],
    selection_mode: Literal["secondary", "pwm_tolerance_then_ligation", "hard_ligation_filter"],
    ligation_applied: bool,
    pwm_effective: bool,
    filtered_candidate_count: int,
) -> str:
    if state == "legacy":
        return "Legacy mode; geometric ranking only because ligation_profile=none."
    if state == "inert":
        return "Ligation-aware scoring is disabled; geometric ranking only."
    if selection_mode == "hard_ligation_filter":
        filter_note = (
            f"Hard ligation filter removed {filtered_candidate_count} candidates before ranking. "
            if filtered_candidate_count > 0
            else "Hard ligation filter admitted the full pool before ranking. "
        )
        if state == "edge_blind":
            return (
                filter_note + "Edge-vs-middle comparison remained unavailable because candidate_positions excludes 0/3."
            )
        return (
            filter_note + "PWM preserved first among surviving candidates."
            if pwm_effective
            else filter_note + "No PWM context; ligation-aware ranking applied among surviving candidates."
        )
    if selection_mode == "pwm_tolerance_then_ligation":
        if not pwm_effective:
            return "No PWM context; tolerance gate unavailable, so ligation-aware ranking applied."
        if state == "edge_blind":
            return (
                "PWM tolerance gate applied before ligation-aware ranking; edge-vs-middle comparison unavailable "
                "because candidate_positions excludes 0/3."
            )
        return "PWM tolerance gate applied before ligation-aware ranking"
    if state == "edge_blind":
        return (
            "PWM preserved first, ligation-aware tie-break applied; edge-vs-middle comparison unavailable "
            "because candidate_positions excludes 0/3."
            if pwm_effective
            else "No PWM context; ligation-aware ranking applied; edge-vs-middle comparison unavailable because "
            "candidate_positions excludes 0/3."
        )
    if ligation_applied and pwm_effective:
        return "PWM preserved first, ligation-aware tie-break applied"
    if ligation_applied:
        return "No PWM context; ligation-aware ranking applied"
    return "Legacy geometric ranking applied"


def build_trace_summary(
    *,
    candidate_count: int,
    trace_sample: OptimizationTraceSample | None,
    trace_len: int,
) -> YiuValidationTraceSummary:
    sample_limit = trace_len if trace_sample is None else trace_sample.sample_limit
    sampled_count = trace_len if trace_sample is None else trace_sample.sampled_count
    truncated = sampled_count < candidate_count
    note = (
        f"Optimizer trace is a bounded sample ({sampled_count}/{candidate_count} candidates, limit={sample_limit})."
        if truncated
        else f"Optimizer trace covers all {candidate_count} candidates."
    )
    return YiuValidationTraceSummary(
        sample_limit=sample_limit,
        sampled_count=sampled_count,
        candidate_count=candidate_count,
        truncated=truncated,
        note=note,
    )


def build_validation_report(
    *,
    spec_name: str,
    normalized: NormalizedPayload,
    bundle_dir: str | None = None,
) -> YiuValidationReport:
    ligation_applied = normalized.chosen_ligation_key is not None
    state, state_note, edge_comparison_available = resolve_ligation_surface_state(normalized)
    ligation_policy = normalized.optimization_decision.ligation_policy
    return YiuValidationReport(
        spec_name=spec_name,
        status="satisfied",
        bundle_dir=bundle_dir,
        ligation=YiuValidationLigationSummary(
            profile=normalized.ligation_profile,
            awareness_mode=normalized.ligation_awareness_mode,
            selection_mode=normalized.ligation_selection_mode,
            applied=ligation_applied,
            state=state,
            state_note=state_note,
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
            chosen_mismatch_classes=[entry.canonical_mismatch_class for entry in normalized.ligation_rationale],
            position_classes=[entry.position_class for entry in normalized.ligation_rationale],
            candidate_count_before_filter=(
                None if ligation_policy is None else ligation_policy.candidate_count_before_filter
            ),
            candidate_count_after_filter=(
                None if ligation_policy is None else ligation_policy.candidate_count_after_filter
            ),
            filtered_candidate_count=None if ligation_policy is None else ligation_policy.filtered_candidate_count,
            decision_note=resolve_ligation_decision_note(
                state=state,
                selection_mode=normalized.ligation_selection_mode,
                ligation_applied=ligation_applied,
                pwm_effective=normalized.motif_context.effective,
                filtered_candidate_count=0 if ligation_policy is None else ligation_policy.filtered_candidate_count,
            ),
        ),
        trace=build_trace_summary(
            candidate_count=normalized.optimization_decision.candidate_count,
            trace_sample=normalized.optimization_decision.trace_sample,
            trace_len=len(normalized.optimization_decision.trace),
        ),
        **payload_summary_from_normalized(normalized),
    )
