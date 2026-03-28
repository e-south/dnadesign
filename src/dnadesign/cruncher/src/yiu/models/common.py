"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/models/common.py

Shared schema primitives and v1 workflow contracts for YIU.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.bio.iupac import normalize_iupac
from dnadesign.cruncher.config.schema_v3 import StrictBaseModel

SequenceMode = Literal["concrete", "iupac_pattern", "pattern"]
ValidationMode = Literal["concrete_realization", "pattern_compatibility"]
CompatibilityStatus = Literal["guaranteed", "possible", "impossible"]
PatternEvidencePolicy = Literal["require_guaranteed", "allow_possible_with_warning"]
RegionProjectionMode = Literal["atomic_required", "compound_allowed", "compound_required"]
WorkflowScope = Literal["core_insert_generation", "insert_plus_backbone_cloning"]
InvariantSpaceKind = Literal[
    "state_sequence",
    "state_duplex",
    "fragment_pool",
    "assembly_junction",
    "compound_retained",
    "topology_state",
]
TopologyKind = Literal[
    "linear_ssdna",
    "linear_dsdna",
    "fragment_pool",
    "annealed_complex",
    "hairpin_ssdna",
    "branched_y",
    "assembly_reaction",
    "circular_dsdna_candidate",
]
PublishedAssemblySpace = Literal[
    "post_ligation",
    "retained_product",
    "adapter_complex",
    "circularized_payload_junction",
]
YIU_PROTOCOL_TEMPLATE_IDS: tuple[str, ...] = (
    "yiu_adapter_hairpin_v1",
    "yiu_circularized_payload_v1",
)
YIU_PROTOCOL_TEMPLATE_ALIASES: dict[str, str] = {
    "msd_hop_retron_eco1_v1": "yiu_adapter_hairpin_v1",
    "yiu_split_payload_circularized_v1": "yiu_circularized_payload_v1",
}
YIU_CANONICAL_TEMPLATE_SUPPORTED_INVARIANT_CLASSES: frozenset[str] = frozenset(
    {
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
    }
)
YIU_V2_STATE_IDS: tuple[str, ...] = (
    "source_oligo_ssdna",
    "source_amplicon_dsdna",
    "post_double_nicking_fragment_pool",
    "post_heat_cleanup_fragment_pool",
    "adapter_annealed_complex",
    "ligated_ssdna_hairpin",
    "hairpin_pcr_linear_insert",
    "post_insert_cleanup_linear_insert",
    "backbone_amplicon",
    "assembly_reaction",
    "assembled_plasmid_candidate",
)
YIU_STEP_KINDS: tuple[str, ...] = (
    "pcr",
    "restriction_digest",
    "circularization",
    "exonuclease_selection",
    "nickase_digest",
    "size_selection",
    "foldback",
    "adapter_ligation",
    "amplification",
)


def _validate_slug(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def normalize_yiu_protocol_template(value: str) -> str:
    normalized = _validate_slug(value, label="protocol_template")
    normalized_template = YIU_PROTOCOL_TEMPLATE_ALIASES.get(normalized, normalized)
    if normalized_template in YIU_PROTOCOL_TEMPLATE_IDS:
        return normalized_template
    allowed = sorted([*YIU_PROTOCOL_TEMPLATE_IDS, *YIU_PROTOCOL_TEMPLATE_ALIASES.keys()])
    raise ValueError(f"protocol_template must be one of {allowed}")


def deprecated_yiu_protocol_template_alias(value: str) -> str | None:
    normalized = str(value or "").strip()
    normalized_template = YIU_PROTOCOL_TEMPLATE_ALIASES.get(normalized)
    if normalized_template is None:
        return None
    warnings.warn(
        f"YIU protocol template alias {normalized!r} is deprecated; use {normalized_template!r}.",
        DeprecationWarning,
        stacklevel=3,
    )
    return normalized


class RegionSpec(StrictBaseModel):
    id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    strand: Literal["primary", "complement", "either"] = "primary"

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="region.id")

    @model_validator(mode="after")
    def _validate_bounds(self) -> "RegionSpec":
        if self.end <= self.start:
            raise ValueError("region.end must be > region.start")
        return self


class PrimerBindingSiteSpec(RegionSpec):
    strand: Literal["primary", "complement"]


class EnzymeSiteSpec(StrictBaseModel):
    id: str
    enzyme: str
    recognition_sequence: str
    start: int = Field(ge=0)
    orientation: Literal["forward", "reverse"] = "forward"
    top_cut_offset: int | None = None
    bottom_cut_offset: int | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="enzyme_site.id")

    @field_validator("enzyme")
    @classmethod
    def _validate_enzyme(cls, value: str) -> str:
        return _validate_slug(value, label="enzyme_site.enzyme")

    @field_validator("recognition_sequence")
    @classmethod
    def _validate_recognition_sequence(cls, value: str) -> str:
        return normalize_iupac(value)

    @model_validator(mode="after")
    def _validate_geometry(self) -> "EnzymeSiteSpec":
        if self.top_cut_offset is None and self.bottom_cut_offset is None:
            raise ValueError("enzyme_site must define at least one cut offset")
        return self

    @property
    def end(self) -> int:
        return self.start + len(self.recognition_sequence)


class SourceOligoSpec(StrictBaseModel):
    sequence: str
    primer_sites: list[PrimerBindingSiteSpec] = Field(default_factory=list)
    restriction_sites: list[EnzymeSiteSpec] = Field(default_factory=list)
    nickase_sites: list[EnzymeSiteSpec] = Field(default_factory=list)
    payload_windows: list[RegionSpec] = Field(default_factory=list)
    homology_windows: list[RegionSpec] = Field(default_factory=list)
    retained_regions: list[RegionSpec] = Field(default_factory=list)
    sacrificial_regions: list[RegionSpec] = Field(default_factory=list)

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_iupac(value)

    @model_validator(mode="after")
    def _validate_annotation_ids(self) -> "SourceOligoSpec":
        seen: set[str] = set()
        for collection in (
            self.primer_sites,
            self.restriction_sites,
            self.nickase_sites,
            self.payload_windows,
            self.homology_windows,
            self.retained_regions,
            self.sacrificial_regions,
        ):
            for item in collection:
                if item.id in seen:
                    raise ValueError(f"duplicate annotation id: {item.id}")
                seen.add(item.id)
                if item.end > len(self.sequence):
                    raise ValueError(f"annotation {item.id} exceeds source_oligo.sequence length")
        return self


class RegionSpecV2(RegionSpec):
    projection_mode: RegionProjectionMode = "atomic_required"
    annotation_class: str | None = None

    @field_validator("annotation_class")
    @classmethod
    def _validate_annotation_class(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_slug(value, label="annotation_class")


class SourceOligoPartInstance(StrictBaseModel):
    id: str
    role: str
    orientation: Literal["forward", "reverse_complement"] = "forward"
    variability: Literal["fixed", "variable"] = "fixed"
    part_id: str | None = None
    sequence: str | None = None
    pattern: str | None = None

    @field_validator("id", "role")
    @classmethod
    def _validate_id_like(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))

    @field_validator("part_id")
    @classmethod
    def _validate_optional_part_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_slug(value, label="part_id")

    @field_validator("sequence", "pattern")
    @classmethod
    def _validate_optional_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return normalize_iupac(value)

    @model_validator(mode="after")
    def _validate_sequence_source(self) -> "SourceOligoPartInstance":
        if self.sequence is None and self.pattern is None:
            raise ValueError("part_instances entries require sequence or pattern")
        return self


class PrimerTailSpec(StrictBaseModel):
    id: str
    primer_binding_core_id: str
    sequence: str

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="primer_tail.id")

    @field_validator("primer_binding_core_id")
    @classmethod
    def _validate_primer_binding_core_id(cls, value: str) -> str:
        return _validate_slug(value, label="primer_tail.primer_binding_core_id")

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_iupac(value)


class OverlapOverride(StrictBaseModel):
    left_annotation_id: str
    right_annotation_id: str
    mode: Literal["allow_nested", "allow_partial", "allow_equal"]
    rationale: str

    @field_validator("left_annotation_id", "right_annotation_id")
    @classmethod
    def _validate_annotation_id(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))

    @field_validator("rationale")
    @classmethod
    def _validate_rationale(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("overlap_override.rationale must be non-empty")
        return text


class SourceOligoAnnotationsV2(StrictBaseModel):
    primer_binding_cores: list[PrimerBindingSiteSpec] = Field(default_factory=list)
    primer_tails: list[PrimerTailSpec] = Field(default_factory=list)
    restriction_sites: list[EnzymeSiteSpec] = Field(default_factory=list)
    nickase_sites: list[EnzymeSiteSpec] = Field(default_factory=list)
    payload_windows: list[RegionSpecV2] = Field(default_factory=list)
    homology_windows: list[RegionSpecV2] = Field(default_factory=list)
    retained_regions: list[RegionSpecV2] = Field(default_factory=list)
    sacrificial_regions: list[RegionSpecV2] = Field(default_factory=list)
    named_regions: list[RegionSpecV2] = Field(default_factory=list)
    overlap_overrides: list[OverlapOverride] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_annotation_ids(self) -> "SourceOligoAnnotationsV2":
        seen: set[str] = set()
        for collection in (
            self.primer_binding_cores,
            self.primer_tails,
            self.restriction_sites,
            self.nickase_sites,
            self.payload_windows,
            self.homology_windows,
            self.retained_regions,
            self.sacrificial_regions,
            self.named_regions,
        ):
            for item in collection:
                if item.id in seen:
                    raise ValueError(f"duplicate annotation id: {item.id}")
                seen.add(item.id)
        primer_core_ids = {item.id for item in self.primer_binding_cores}
        for tail in self.primer_tails:
            if tail.primer_binding_core_id not in primer_core_ids:
                raise ValueError(
                    f"primer tail {tail.id} references unknown primer_binding_core_id {tail.primer_binding_core_id!r}"
                )
        for override in self.overlap_overrides:
            if override.left_annotation_id not in seen:
                raise ValueError(
                    "YIU_OVERLAP_OVERRIDE_REF_UNKNOWN: overlap override references unknown "
                    f"annotation {override.left_annotation_id!r}"
                )
            if override.right_annotation_id not in seen:
                raise ValueError(
                    "YIU_OVERLAP_OVERRIDE_REF_UNKNOWN: overlap override references unknown "
                    f"annotation {override.right_annotation_id!r}"
                )
        return self


class SourceOligoSpecV2(StrictBaseModel):
    sequence: str | None = None
    authored_sequence: str | None = None
    part_instances: list[SourceOligoPartInstance] = Field(default_factory=list)
    annotations: SourceOligoAnnotationsV2

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return normalize_iupac(value)

    @model_validator(mode="after")
    def _normalize_sequence(self) -> "SourceOligoSpecV2":
        normalized_sequence = self.sequence
        if normalized_sequence is None and self.authored_sequence is not None:
            normalized_sequence = normalize_iupac(self.authored_sequence)
        if normalized_sequence is None and self.part_instances:
            normalized_sequence = "".join(
                instance.sequence or instance.pattern or "" for instance in self.part_instances
            )
        if normalized_sequence is None:
            raise ValueError("source_oligo requires sequence, authored_sequence, or part_instances")
        if self.sequence is not None and self.authored_sequence is not None:
            authored_normalized = normalize_iupac(self.authored_sequence)
            if self.sequence != authored_normalized:
                raise ValueError("source_oligo.sequence must match authored_sequence after normalization")
        self.sequence = normalized_sequence
        return self

    @model_validator(mode="after")
    def _validate_annotation_bounds(self) -> "SourceOligoSpecV2":
        if self.sequence is None:
            raise ValueError("source_oligo.sequence normalization failed")
        sequence_length = len(self.sequence)
        for collection in (
            self.annotations.primer_binding_cores,
            self.annotations.restriction_sites,
            self.annotations.nickase_sites,
            self.annotations.payload_windows,
            self.annotations.homology_windows,
            self.annotations.retained_regions,
            self.annotations.sacrificial_regions,
            self.annotations.named_regions,
        ):
            for item in collection:
                if item.end > sequence_length:
                    raise ValueError(f"annotation {item.id} exceeds source_oligo.sequence length")
        return self


class YiuStepSpec(StrictBaseModel):
    kind: Literal[
        "pcr",
        "restriction_digest",
        "circularization",
        "exonuclease_selection",
        "nickase_digest",
        "size_selection",
        "foldback",
        "adapter_ligation",
        "amplification",
    ]
    id: str
    forward_primer_site: str | None = None
    reverse_primer_site: str | None = None
    left_site: str | None = None
    right_site: str | None = None
    expected_left_overhang: str | None = None
    expected_right_overhang: str | None = None
    compatibility: Literal["exact_complement", "partial_complement", "bulged"] | None = None
    site_ids: list[str] = Field(default_factory=list)
    sacrificial_region_ids: list[str] = Field(default_factory=list)
    retained_region_ids: list[str] = Field(default_factory=list)
    left_homology_window: str | None = None
    right_homology_window: str | None = None
    min_complementary_bases: int | None = Field(default=None, ge=0)
    min_paired_nt: int | None = Field(default=None, ge=0)
    max_unpaired_tail_nt: int | None = Field(default=None, ge=0)
    max_bulge_nt: int | None = Field(default=None, ge=0)
    adapter_sequence: str | None = None
    forward_primer_requirement: str | None = None
    reverse_primer_requirement: str | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="step.id")

    @field_validator("expected_left_overhang", "expected_right_overhang", "adapter_sequence")
    @classmethod
    def _validate_iupac_values(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return normalize_iupac(value)

    @field_validator("forward_primer_requirement", "reverse_primer_requirement")
    @classmethod
    def _validate_primer_requirement(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return normalize_iupac(value)

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> "YiuStepSpec":
        required_fields: dict[str, tuple[str, ...]] = {
            "pcr": ("forward_primer_site", "reverse_primer_site"),
            "restriction_digest": ("left_site", "right_site", "expected_left_overhang", "expected_right_overhang"),
            "circularization": ("compatibility",),
            "nickase_digest": ("site_ids", "sacrificial_region_ids", "retained_region_ids"),
            "foldback": ("left_homology_window", "right_homology_window", "min_complementary_bases"),
            "amplification": ("forward_primer_requirement", "reverse_primer_requirement"),
        }
        for field_name in required_fields.get(self.kind, ()):
            value = getattr(self, field_name)
            if value is None or value == []:
                raise ValueError(f"step {self.id} ({self.kind}) requires field {field_name}")
        return self


class StepGraphSpec(StrictBaseModel):
    steps: list[YiuStepSpec]

    @model_validator(mode="after")
    def _validate_canonical_order(self) -> "StepGraphSpec":
        kinds = [step.kind for step in self.steps]
        if kinds != list(YIU_STEP_KINDS):
            raise ValueError(f"step_graph.steps must use the canonical YIU order: {', '.join(YIU_STEP_KINDS)}")
        ids = [step.id for step in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("step_graph.steps ids must be unique")
        return self


class PayloadGoalSpec(StrictBaseModel):
    assembled_payload: str
    left_half_ref: str
    right_half_ref: str
    junction_rule: Literal["contiguous_after_ligation"]

    @field_validator("assembled_payload")
    @classmethod
    def _validate_payload(cls, value: str) -> str:
        return normalize_iupac(value)


class LinearDepletionPolicy(StrictBaseModel):
    enabled: bool = True
    enzyme: str | None = None


class SizeSelectionPolicy(StrictBaseModel):
    min_removed_fragment_nt: int | None = Field(default=None, ge=0)
    max_retained_sacrificial_fragment_nt: int | None = Field(default=None, ge=0)
    min_retained_product_nt: int | None = Field(default=None, ge=0)


class CleanupPolicySpec(StrictBaseModel):
    linear_depletion: LinearDepletionPolicy = Field(default_factory=LinearDepletionPolicy)
    size_selection: SizeSelectionPolicy = Field(default_factory=SizeSelectionPolicy)


class PrimerBindingRequirementSpec(StrictBaseModel):
    id: str
    sequence: str

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="primer_binding_requirement.id")

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_iupac(value)


class AdapterPolicySpec(StrictBaseModel):
    y_adapter_id: str | None = None
    adapter_sequence: str | None = None
    primer_binding_requirements: list[PrimerBindingRequirementSpec] = Field(default_factory=list)

    @field_validator("adapter_sequence")
    @classmethod
    def _validate_adapter_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return normalize_iupac(value)


class CatalogRefs(StrictBaseModel):
    restriction_enzymes: Path | None = None
    nickases: Path | None = None
    adapters: Path | None = None
