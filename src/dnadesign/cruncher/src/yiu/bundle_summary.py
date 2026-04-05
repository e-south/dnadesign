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
from dnadesign.cruncher.yiu.bundle_models import PayloadVisualInventory
from dnadesign.cruncher.yiu.domain_models import JunctionSelection, MismatchSelection, NormalizedPayload


class YiuSplitPayloadSummary(StrictBaseModel):
    left_payload_body_sequence_5to3: str
    selected_sticky_end_sequence_5to3: str
    canonical_sticky_end_sequence_5to3: str
    right_payload_body_sequence_5to3: str


class YiuSequenceSummary(StrictBaseModel):
    reference_payload_sequence_5to3: str
    reference_complement_sequence_3to5: str
    selected_payload_sequence_5to3: str
    selected_complement_sequence_3to5: str
    junction_payload_sequence_5to3: str
    junction_complement_sequence_3to5: str
    split_payload: YiuSplitPayloadSummary


class YiuPwmSummary(StrictBaseModel):
    mode: Literal["none", "use_if_available", "require"]
    effective: bool
    motif_count: int = Field(ge=0)
    worst_loss: float = Field(ge=0.0)
    total_loss: float = Field(ge=0.0)


class YiuBundleSummary(StrictBaseModel):
    summary_contract: Literal["yiu_bundle_summary_v1"] = "yiu_bundle_summary_v1"
    schema_version: Literal[1] = 1
    spec_name: str
    payload_label: str | None = None
    input_kind: Literal["user_sequence", "sample_hit"]
    payload_length: int = Field(ge=1)
    sequence_summary: YiuSequenceSummary
    junction: JunctionSelection
    mismatches: list[MismatchSelection] = Field(default_factory=list)
    pwm: YiuPwmSummary
    view_ids: list[str] = Field(default_factory=list)
    render_status: Literal["not_requested", "rendered", "missing", "partial", "failed"] = "not_requested"


def build_sequence_summary(normalized: NormalizedPayload) -> YiuSequenceSummary:
    junction_start = normalized.junction.start
    junction_end = normalized.junction.end
    return YiuSequenceSummary(
        reference_payload_sequence_5to3=normalized.reference_payload_sequence,
        reference_complement_sequence_3to5=normalized.reference_complement_sequence,
        selected_payload_sequence_5to3=normalized.selected_payload_sequence,
        selected_complement_sequence_3to5=normalized.selected_complement_sequence,
        junction_payload_sequence_5to3=normalized.selected_payload_sequence[junction_start:junction_end],
        junction_complement_sequence_3to5=normalized.selected_complement_sequence[junction_start:junction_end],
        split_payload=YiuSplitPayloadSummary(
            left_payload_body_sequence_5to3=normalized.selected_payload_sequence[:junction_start],
            selected_sticky_end_sequence_5to3=normalized.selected_complement_sequence[junction_start:junction_end][
                ::-1
            ],
            canonical_sticky_end_sequence_5to3=normalized.reference_complement_sequence[junction_start:junction_end][
                ::-1
            ],
            right_payload_body_sequence_5to3=normalized.selected_payload_sequence[junction_end:],
        ),
    )


def build_bundle_summary(
    *,
    normalized: NormalizedPayload,
    inventory: PayloadVisualInventory,
) -> YiuBundleSummary:
    return YiuBundleSummary(
        spec_name=inventory.spec_name,
        payload_label=normalized.payload_label,
        input_kind=normalized.input_kind,
        payload_length=normalized.payload_length,
        sequence_summary=build_sequence_summary(normalized),
        junction=normalized.junction,
        mismatches=normalized.mismatches,
        pwm=YiuPwmSummary(
            mode=normalized.motif_context.requested_mode,
            effective=normalized.motif_context.effective,
            motif_count=len(normalized.motif_context.motifs),
            worst_loss=normalized.optimization_decision.winner.worst_loss,
            total_loss=normalized.optimization_decision.winner.total_loss,
        ),
        view_ids=[view.view_id for view in inventory.views],
        render_status=inventory.render_status,
    )


__all__ = [
    "YiuBundleSummary",
    "YiuPwmSummary",
    "YiuSequenceSummary",
    "YiuSplitPayloadSummary",
    "build_bundle_summary",
    "build_sequence_summary",
]
