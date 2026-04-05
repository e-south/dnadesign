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
from dnadesign.cruncher.yiu.domain_models import JunctionSelection, MismatchSelection, NormalizedPayload

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


def build_validation_report(
    *,
    spec_name: str,
    normalized: NormalizedPayload,
    bundle_dir: str | None = None,
) -> YiuValidationReport:
    return YiuValidationReport(
        spec_name=spec_name,
        status="satisfied",
        bundle_dir=bundle_dir,
        **payload_summary_from_normalized(normalized),
    )
