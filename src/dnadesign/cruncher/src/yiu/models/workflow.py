"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/models/workflow.py

Top-level v1 and v2 YIU workflow schema documents.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.models.catalogs import LigationRuleSpec, OutputSpec, OutputSpecV2
from dnadesign.cruncher.yiu.models.common import (
    YIU_CANONICAL_TEMPLATE_SUPPORTED_INVARIANT_CLASSES,
    AdapterPolicySpec,
    CatalogRefs,
    CleanupPolicySpec,
    PayloadGoalSpec,
    SourceOligoSpec,
    SourceOligoSpecV2,
    StepGraphSpec,
    WorkflowScope,
    _validate_slug,
    normalize_yiu_protocol_template,
)
from dnadesign.cruncher.yiu.models.v2_steps import (
    CatalogRefsV2,
    CompoundRegionRef,
    PayloadGoalSpecV2,
    YiuHardInvariant,
    YiuStepsSpecV2,
    YiuTemplateBindingsV2,
)


class YiuProcessSpec(StrictBaseModel):
    schema_version: int = 1
    protocol: Literal["yiu_v1"] = "yiu_v1"
    name: str
    source_oligo: SourceOligoSpec
    step_graph: StepGraphSpec
    payload_goal: PayloadGoalSpec
    cleanup_policy: CleanupPolicySpec = Field(default_factory=CleanupPolicySpec)
    adapter_policy: AdapterPolicySpec = Field(default_factory=AdapterPolicySpec)
    catalogs: CatalogRefs = Field(default_factory=CatalogRefs)
    output: OutputSpec = Field(default_factory=OutputSpec)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if int(value) != 1:
            raise ValueError("yiu.schema_version must be 1")
        return int(value)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _validate_slug(value, label="yiu.name")

    @model_validator(mode="after")
    def _validate_cross_step_contracts(self) -> "YiuProcessSpec":
        adapter_step = next((step for step in self.step_graph.steps if step.kind == "adapter_ligation"), None)
        if adapter_step is not None and (
            adapter_step.adapter_sequence is None
            and self.adapter_policy.adapter_sequence is None
            and self.adapter_policy.y_adapter_id is None
        ):
            raise ValueError(
                "adapter_ligation requires an adapter sequence source from "
                "step.adapter_sequence, adapter_policy.adapter_sequence, or adapter_policy.y_adapter_id"
            )
        return self


class YiuProcessSpecV2(StrictBaseModel):
    schema_version: int = 2
    family: Literal["yiu"] = "yiu"
    protocol_template: str = "yiu_adapter_hairpin_v1"
    workflow_scope: WorkflowScope = "core_insert_generation"
    name: str
    source_oligo: SourceOligoSpecV2
    steps: YiuStepsSpecV2
    payload_goal: PayloadGoalSpecV2
    ligation_rule: LigationRuleSpec | None = None
    template_bindings: YiuTemplateBindingsV2 | None = None
    compound_regions: list[CompoundRegionRef] = Field(default_factory=list)
    hard_invariants: list[YiuHardInvariant] = Field(default_factory=list)
    catalogs: CatalogRefsV2 = Field(default_factory=CatalogRefsV2)
    output: OutputSpecV2 = Field(default_factory=OutputSpecV2)
    template_alias_used: str | None = Field(default=None, exclude=True)
    template_alias_status: Literal["deprecated_alias"] | None = Field(default=None, exclude=True)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if int(value) != 2:
            raise ValueError("yiu.schema_version must be 2")
        return int(value)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _validate_slug(value, label="yiu.name")

    @field_validator("protocol_template")
    @classmethod
    def _validate_protocol_template(cls, value: str) -> str:
        return normalize_yiu_protocol_template(value)

    @model_validator(mode="after")
    def _validate_v2_contracts(self) -> "YiuProcessSpecV2":
        annotations = self.source_oligo.annotations
        primer_core_ids = {core.id for core in annotations.primer_binding_cores}
        restriction_site_ids = {site.id for site in annotations.restriction_sites}
        region_ids = {
            region.id
            for collection in (
                annotations.payload_windows,
                annotations.homology_windows,
                annotations.retained_regions,
                annotations.sacrificial_regions,
                annotations.named_regions,
            )
            for region in collection
        }
        if self.payload_goal.left_half_ref not in region_ids:
            raise ValueError(
                f"payload_goal.left_half_ref references unknown region {self.payload_goal.left_half_ref!r}"
            )
        if self.payload_goal.right_half_ref not in region_ids:
            raise ValueError(
                f"payload_goal.right_half_ref references unknown region {self.payload_goal.right_half_ref!r}"
            )
        if self.workflow_scope == "core_insert_generation":
            if self.steps.backbone_pcr.enabled or self.steps.golden_gate_assembly.enabled:
                raise ValueError(
                    "backbone_pcr and golden_gate_assembly must stay disabled for workflow_scope=core_insert_generation"
                )
        if self.protocol_template == "yiu_circularized_payload_v1":
            required_split_fields = {
                "type_iis_digest": self.steps.type_iis_digest,
                "circularization": self.steps.circularization,
                "sacrificial_digest": self.steps.sacrificial_digest,
                "snapback_adapter_engagement": self.steps.snapback_adapter_engagement,
            }
            for field_name, value in required_split_fields.items():
                if value is None:
                    raise ValueError(
                        f"steps.{field_name} is required for protocol_template=yiu_circularized_payload_v1"
                    )
            if self.output.publish_contract_version != 3:
                raise ValueError(
                    "output.publish_contract_version must be 3 for protocol_template=yiu_circularized_payload_v1"
                )
            if self.template_bindings is None:
                raise ValueError(
                    "YIU_TEMPLATE_BINDING_MISSING: template_bindings is required for "
                    "protocol_template=yiu_circularized_payload_v1"
                )
            binding_checks = (
                (
                    "source_forward_primer_core_ref",
                    self.template_bindings.source_forward_primer_core_ref,
                    primer_core_ids,
                ),
                (
                    "source_reverse_primer_core_ref",
                    self.template_bindings.source_reverse_primer_core_ref,
                    primer_core_ids,
                ),
                ("snapback_seed_region_ref", self.template_bindings.snapback_seed_region_ref, region_ids),
                ("retained_left_region_ref", self.template_bindings.retained_left_region_ref, region_ids),
                ("retained_right_region_ref", self.template_bindings.retained_right_region_ref, region_ids),
                (
                    "circularization_left_overhang_ref",
                    self.template_bindings.circularization_left_overhang_ref,
                    restriction_site_ids,
                ),
                (
                    "circularization_right_overhang_ref",
                    self.template_bindings.circularization_right_overhang_ref,
                    restriction_site_ids,
                ),
            )
            for field_name, ref, allowed_refs in binding_checks:
                if ref not in allowed_refs:
                    raise ValueError(
                        "YIU_TEMPLATE_BINDING_REF_UNKNOWN: "
                        f"template_bindings.{field_name} references unknown id {ref!r}"
                    )
            for ref in self.template_bindings.primary_sacrificial_region_refs:
                if ref not in region_ids:
                    raise ValueError(
                        "YIU_TEMPLATE_BINDING_REF_UNKNOWN: template_bindings.primary_sacrificial_region_refs "
                        f"references unknown region {ref!r}"
                    )
        elif self.template_bindings is not None:
            raise ValueError(
                "YIU_INVARIANT_CLASS_NOT_ALLOWED_FOR_TEMPLATE: template_bindings is only supported for "
                "protocol_template=yiu_circularized_payload_v1"
            )
        for compound_region in self.compound_regions:
            for segment in compound_region.segments:
                if segment.source_region_ref not in region_ids:
                    raise ValueError(
                        f"compound_regions.{compound_region.id} references unknown region {segment.source_region_ref!r}"
                    )
        supported_invariant_classes = (
            YIU_CANONICAL_TEMPLATE_SUPPORTED_INVARIANT_CLASSES
            if self.protocol_template == "yiu_circularized_payload_v1"
            else frozenset()
        )
        compound_region_ids = {region.id for region in self.compound_regions}
        for invariant in self.hard_invariants:
            if invariant.class_ == "cloning_geometry" and self.workflow_scope == "core_insert_generation":
                raise ValueError(
                    "YIU_INVARIANT_CLASS_NOT_ALLOWED_FOR_SCOPE: hard_invariants "
                    f"{invariant.id} class cloning_geometry is unsupported for workflow_scope=core_insert_generation"
                )
            if invariant.class_ == "cloning_geometry":
                raise ValueError(
                    f"YIU_UNSUPPORTED_INVARIANT_CLASS: hard_invariants {invariant.id} class cloning_geometry "
                    "is not implemented in this tranche"
                )
            if invariant.class_ not in supported_invariant_classes:
                code = (
                    "YIU_INVARIANT_CLASS_NOT_ALLOWED_FOR_TEMPLATE"
                    if self.protocol_template != "yiu_circularized_payload_v1"
                    else "YIU_UNSUPPORTED_INVARIANT_CLASS"
                )
                raise ValueError(
                    f"{code}: hard_invariants {invariant.id} class {invariant.class_} is not supported for "
                    f"protocol_template={self.protocol_template}"
                )
            if (
                invariant.region_ref is not None
                and invariant.region_ref not in region_ids
                and invariant.region_ref not in compound_region_ids
            ):
                raise ValueError(f"hard_invariants.{invariant.id} references unknown region {invariant.region_ref!r}")
        return self


class YiuSpecDocument(StrictBaseModel):
    yiu: YiuProcessSpec


class YiuSpecDocumentV2(StrictBaseModel):
    yiu: YiuProcessSpecV2
