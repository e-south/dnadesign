"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/models/v2_steps.py

Schema contracts for YIU v2 steps, template bindings, and hard invariants.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.bio.iupac import normalize_iupac
from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.models.catalogs import (
    BulgedCompatibilityRule,
    LigationRuleSpec,
    PartialComplementRule,
)
from dnadesign.cruncher.yiu.models.common import (
    InvariantSpaceKind,
    PatternEvidencePolicy,
    PublishedAssemblySpace,
    _validate_slug,
)

LigationCompatibilityMode = Literal["exact_complement", "partial_complement", "bulged"]


class SourcePcrStepSpecV2(StrictBaseModel):
    forward_primer_id: str
    reverse_primer_id: str

    @field_validator("forward_primer_id", "reverse_primer_id")
    @classmethod
    def _validate_primer_id(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))


class DoubleNickingDigestStepSpecV2(StrictBaseModel):
    enzymes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_enzymes(self) -> "DoubleNickingDigestStepSpecV2":
        if not self.enzymes:
            raise ValueError("double_nicking_digest.enzymes must be non-empty")
        return self


class HeatCleanupStepSpecV2(StrictBaseModel):
    enabled: bool = True
    min_retained_nt: int | None = Field(default=None, ge=0)


class AdapterAnnealStepSpecV2(StrictBaseModel):
    adapter_id: str
    compatibility_mode: LigationCompatibilityMode = "exact_complement"
    partial_complement: PartialComplementRule | None = None
    bulged: BulgedCompatibilityRule | None = None
    ligation_rule: LigationRuleSpec | None = None

    @field_validator("adapter_id")
    @classmethod
    def _validate_adapter_id(cls, value: str) -> str:
        return _validate_slug(value, label="adapter_anneal.adapter_id")


class HairpinLigationStepSpecV2(StrictBaseModel):
    ligase: str | None = None
    require_5p_phosphate: bool = False
    compatibility_mode: LigationCompatibilityMode = "exact_complement"
    partial_complement: PartialComplementRule | None = None
    bulged: BulgedCompatibilityRule | None = None
    ligation_rule: LigationRuleSpec | None = None


class TypeIisDigestStepSpecV2(StrictBaseModel):
    enzyme_id: str
    site_ids: list[str] = Field(default_factory=list)

    @field_validator("enzyme_id")
    @classmethod
    def _validate_enzyme_id(cls, value: str) -> str:
        return _validate_slug(value, label="type_iis_digest.enzyme_id")

    @model_validator(mode="after")
    def _validate_site_ids(self) -> "TypeIisDigestStepSpecV2":
        if not self.site_ids:
            raise ValueError("type_iis_digest.site_ids must be non-empty")
        return self


class CircularizationStepSpecV2(StrictBaseModel):
    ligation_rule: LigationRuleSpec


class ExonucleaseCleanupStepSpecV2(StrictBaseModel):
    enabled: bool = True
    enzyme: str | None = None


class SacrificialDigestStepSpecV2(StrictBaseModel):
    enzyme_ids: list[str] = Field(default_factory=list)
    site_ids: list[str] = Field(default_factory=list)
    sacrificial_region_ids: list[str] = Field(default_factory=list)
    retained_region_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_lists(self) -> "SacrificialDigestStepSpecV2":
        if not self.enzyme_ids:
            raise ValueError("sacrificial_digest.enzyme_ids must be non-empty")
        if not self.site_ids:
            raise ValueError("sacrificial_digest.site_ids must be non-empty")
        if not self.sacrificial_region_ids:
            raise ValueError("sacrificial_digest.sacrificial_region_ids must be non-empty")
        if not self.retained_region_ids:
            raise ValueError("sacrificial_digest.retained_region_ids must be non-empty")
        return self


class FragmentCleanupStepSpecV2(StrictBaseModel):
    enabled: bool = True
    max_fragment_nt: int | None = Field(default=None, ge=0)
    min_retained_nt: int | None = Field(default=None, ge=0)


class SnapbackAdapterEngagementStepSpecV2(StrictBaseModel):
    adapter_id: str
    ligation_rule: LigationRuleSpec

    @field_validator("adapter_id")
    @classmethod
    def _validate_adapter_id(cls, value: str) -> str:
        return _validate_slug(value, label="snapback_adapter_engagement.adapter_id")


class SinglePrimerPrecyclesSpecV2(StrictBaseModel):
    enabled: bool = False
    primer_id: str | None = None
    cycles: int | None = Field(default=None, ge=1)

    @field_validator("primer_id")
    @classmethod
    def _validate_primer_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_slug(value, label="single_primer_precycles.primer_id")

    @model_validator(mode="after")
    def _validate_enabled_fields(self) -> "SinglePrimerPrecyclesSpecV2":
        if self.enabled and (self.primer_id is None or self.cycles is None):
            raise ValueError("single_primer_precycles requires primer_id and cycles when enabled")
        return self


class XStructureResolutionCycleSpecV2(StrictBaseModel):
    enabled: bool = False


class HairpinPcrStepSpecV2(StrictBaseModel):
    forward_primer_id: str
    reverse_primer_id: str
    single_primer_precycles: SinglePrimerPrecyclesSpecV2 = Field(default_factory=SinglePrimerPrecyclesSpecV2)
    x_structure_resolution_cycle: XStructureResolutionCycleSpecV2 = Field(
        default_factory=XStructureResolutionCycleSpecV2
    )

    @field_validator("forward_primer_id", "reverse_primer_id")
    @classmethod
    def _validate_primer_id(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))


class InsertCleanupStepSpecV2(StrictBaseModel):
    enabled: bool = False
    method: str | None = None


class BackbonePcrStepSpecV2(StrictBaseModel):
    enabled: bool = False
    forward_primer_id: str | None = None
    reverse_primer_id: str | None = None
    backbone_id: str | None = None

    @field_validator("forward_primer_id", "reverse_primer_id", "backbone_id")
    @classmethod
    def _validate_optional_ids(cls, value: str | None, info) -> str | None:
        if value is None:
            return value
        return _validate_slug(value, label=str(info.field_name))


class GoldenGateAssemblyStepSpecV2(StrictBaseModel):
    enabled: bool = False
    enzyme: str | None = None
    backbone_id: str | None = None

    @field_validator("enzyme", "backbone_id")
    @classmethod
    def _validate_optional_ids(cls, value: str | None, info) -> str | None:
        if value is None:
            return value
        return _validate_slug(value, label=str(info.field_name))


class YiuStepsSpecV2(StrictBaseModel):
    source_pcr: SourcePcrStepSpecV2
    double_nicking_digest: DoubleNickingDigestStepSpecV2 | None = None
    heat_cleanup: HeatCleanupStepSpecV2 = Field(default_factory=HeatCleanupStepSpecV2)
    adapter_anneal: AdapterAnnealStepSpecV2 | None = None
    type_iis_digest: TypeIisDigestStepSpecV2 | None = None
    circularization: CircularizationStepSpecV2 | None = None
    exonuclease_cleanup: ExonucleaseCleanupStepSpecV2 = Field(default_factory=ExonucleaseCleanupStepSpecV2)
    sacrificial_digest: SacrificialDigestStepSpecV2 | None = None
    fragment_cleanup: FragmentCleanupStepSpecV2 = Field(default_factory=FragmentCleanupStepSpecV2)
    snapback_adapter_engagement: SnapbackAdapterEngagementStepSpecV2 | None = None
    hairpin_ligation: HairpinLigationStepSpecV2
    hairpin_pcr: HairpinPcrStepSpecV2
    insert_cleanup: InsertCleanupStepSpecV2 = Field(default_factory=InsertCleanupStepSpecV2)
    backbone_pcr: BackbonePcrStepSpecV2 = Field(default_factory=BackbonePcrStepSpecV2)
    golden_gate_assembly: GoldenGateAssemblyStepSpecV2 = Field(default_factory=GoldenGateAssemblyStepSpecV2)


class PayloadGoalSpecV2(StrictBaseModel):
    assembled_payload_pattern: str
    left_half_ref: str
    right_half_ref: str
    assembly_space: PublishedAssemblySpace = "post_ligation"
    evidence_policy: PatternEvidencePolicy = "require_guaranteed"

    @field_validator("assembled_payload_pattern")
    @classmethod
    def _validate_payload_pattern(cls, value: str) -> str:
        return normalize_iupac(value)

    @field_validator("left_half_ref", "right_half_ref")
    @classmethod
    def _validate_refs(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))


class CatalogRefsV2(StrictBaseModel):
    enzymes: Path | None = None
    oligo_parts: Path | None = None
    backbones: Path | None = None


class CompoundRegionSegmentRef(StrictBaseModel):
    source_state: str
    source_region_ref: str
    orientation: Literal["forward", "reverse_complement"] = "forward"

    @field_validator("source_state", "source_region_ref")
    @classmethod
    def _validate_id_like(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))


class CompoundRegionRef(StrictBaseModel):
    id: str
    segments: list[CompoundRegionSegmentRef] = Field(default_factory=list)
    join_policy: Literal["concatenate", "junction_assemble"] = "concatenate"

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="compound_region.id")

    @model_validator(mode="after")
    def _validate_segments(self) -> "CompoundRegionRef":
        if not self.segments:
            raise ValueError("compound_region.segments must be non-empty")
        return self


class YiuTemplateBindingsV2(StrictBaseModel):
    source_forward_primer_core_ref: str
    source_reverse_primer_core_ref: str
    snapback_seed_region_ref: str
    retained_left_region_ref: str
    retained_right_region_ref: str
    primary_sacrificial_region_refs: list[str] = Field(default_factory=list)
    circularization_left_overhang_ref: str
    circularization_right_overhang_ref: str

    @field_validator(
        "source_forward_primer_core_ref",
        "source_reverse_primer_core_ref",
        "snapback_seed_region_ref",
        "retained_left_region_ref",
        "retained_right_region_ref",
        "circularization_left_overhang_ref",
        "circularization_right_overhang_ref",
    )
    @classmethod
    def _validate_ref_fields(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))

    @field_validator("primary_sacrificial_region_refs")
    @classmethod
    def _validate_primary_sacrificial_region_refs(cls, value: list[str]) -> list[str]:
        refs = [_validate_slug(item, label="primary_sacrificial_region_refs") for item in value]
        if not refs:
            raise ValueError("YIU_TEMPLATE_BINDING_MISSING: primary_sacrificial_region_refs must be non-empty")
        return refs


class YiuHardInvariant(StrictBaseModel):
    id: str
    class_: Literal[
        "region_pattern",
        "enzyme_site",
        "cut_geometry",
        "ligation_compatibility",
        "payload_assembly",
        "retained_survival",
        "sacrificial_fragmentation",
        "snapback_exposure",
        "adapter_binding",
        "primer_binding",
        "cloning_geometry",
    ] = Field(alias="class")
    state_ref: str | None = None
    transform_ref: str | None = None
    space_kind: InvariantSpaceKind
    strand_scope: Literal["primary", "complement", "either", "both"] | None = None
    region_ref: str | None = None
    evidence_policy: Literal["require_guaranteed"] = "require_guaranteed"
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="hard_invariant.id")

    @field_validator("state_ref", "transform_ref", "region_ref")
    @classmethod
    def _validate_optional_refs(cls, value: str | None, info) -> str | None:
        if value is None:
            return value
        return _validate_slug(value, label=str(info.field_name))
