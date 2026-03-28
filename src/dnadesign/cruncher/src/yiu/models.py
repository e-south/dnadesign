"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/models.py

Strict schema and report contracts for the YIU workflow family.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.bio.iupac import normalize_iupac
from dnadesign.cruncher.config.schema_v3 import StrictBaseModel

SequenceMode = Literal["concrete", "iupac_pattern", "pattern"]
ValidationMode = Literal["concrete_realization", "pattern_compatibility"]
CompatibilityStatus = Literal["guaranteed", "possible", "impossible"]
PatternEvidencePolicy = Literal["require_guaranteed", "allow_possible_with_warning"]
RegionProjectionMode = Literal["atomic_required", "compound_allowed", "compound_required"]
LigationCompatibilityMode = Literal["exact_complement", "partial_complement", "bulged"]
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
    canonical = YIU_PROTOCOL_TEMPLATE_ALIASES.get(normalized, normalized)
    if canonical in YIU_PROTOCOL_TEMPLATE_IDS:
        return canonical
    allowed = sorted([*YIU_PROTOCOL_TEMPLATE_IDS, *YIU_PROTOCOL_TEMPLATE_ALIASES.keys()])
    raise ValueError(f"protocol_template must be one of {allowed}")


def deprecated_yiu_protocol_template_alias(value: str) -> str | None:
    normalized = str(value or "").strip()
    canonical = YIU_PROTOCOL_TEMPLATE_ALIASES.get(normalized)
    if canonical is None:
        return None
    warnings.warn(
        f"YIU protocol template alias {normalized!r} is deprecated; use {canonical!r}.",
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
                end = item.end if isinstance(item, RegionSpec) else item.end
                if end > len(self.sequence):
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
        canonical_sequence = self.sequence
        if canonical_sequence is None and self.authored_sequence is not None:
            canonical_sequence = normalize_iupac(self.authored_sequence)
        if canonical_sequence is None and self.part_instances:
            canonical_sequence = "".join(
                instance.sequence or instance.pattern or "" for instance in self.part_instances
            )
        if canonical_sequence is None:
            raise ValueError("source_oligo requires sequence, authored_sequence, or part_instances")
        if self.sequence is not None and self.authored_sequence is not None:
            authored_normalized = normalize_iupac(self.authored_sequence)
            if self.sequence != authored_normalized:
                raise ValueError("source_oligo.sequence must match authored_sequence after normalization")
        self.sequence = canonical_sequence
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


class YiuEnzymeCatalogEntry(StrictBaseModel):
    id: str
    recognition_sequence: str
    top_cut_offset: int | None = None
    bottom_cut_offset: int | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="catalog_entry.id")

    @field_validator("recognition_sequence")
    @classmethod
    def _validate_recognition_sequence(cls, value: str) -> str:
        return normalize_iupac(value)


class YiuEnzymeCatalogSpec(StrictBaseModel):
    entries: list[YiuEnzymeCatalogEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> "YiuEnzymeCatalogSpec":
        ids = [entry.id for entry in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("catalog enzyme ids must be unique")
        return self


class YiuRestrictionCatalogDocument(StrictBaseModel):
    restriction_enzymes: YiuEnzymeCatalogSpec


class YiuNickaseCatalogDocument(StrictBaseModel):
    nickases: YiuEnzymeCatalogSpec


class YiuAdapterCatalogEntry(StrictBaseModel):
    id: str
    sequence: str

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="adapter_catalog_entry.id")

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_iupac(value)


class YiuAdapterCatalogSpec(StrictBaseModel):
    entries: list[YiuAdapterCatalogEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> "YiuAdapterCatalogSpec":
        ids = [entry.id for entry in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("catalog adapter ids must be unique")
        return self


class YiuAdapterCatalogDocument(StrictBaseModel):
    adapters: YiuAdapterCatalogSpec


class YiuOligoPartCatalogEntry(StrictBaseModel):
    id: str
    part_kind: Literal["primer", "adapter", "backbone", "other"] = "other"
    sequence: str
    phosphorylated_5p: bool = False

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="oligo_part_catalog_entry.id")

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_iupac(value)


class YiuOligoPartCatalogSpec(StrictBaseModel):
    entries: list[YiuOligoPartCatalogEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> "YiuOligoPartCatalogSpec":
        ids = [entry.id for entry in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("catalog oligo-part ids must be unique")
        return self


class YiuOligoPartCatalogDocument(StrictBaseModel):
    oligo_parts: YiuOligoPartCatalogSpec


class YiuBackboneCatalogEntry(StrictBaseModel):
    id: str
    sequence: str | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="backbone_catalog_entry.id")

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return normalize_iupac(value)


class YiuBackboneCatalogSpec(StrictBaseModel):
    entries: list[YiuBackboneCatalogEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> "YiuBackboneCatalogSpec":
        ids = [entry.id for entry in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("catalog backbone ids must be unique")
        return self


class YiuBackboneCatalogDocument(StrictBaseModel):
    backbones: YiuBackboneCatalogSpec


class YiuGenericEnzymeCatalogDocument(StrictBaseModel):
    enzymes: YiuEnzymeCatalogSpec


class OutputSpec(StrictBaseModel):
    run_dir: Path = Path("outputs/yiu/explicit")
    emit_view_contracts: bool = True

    @field_validator("run_dir")
    @classmethod
    def _validate_run_dir(cls, value: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            raise ValueError("output.run_dir must be relative to the workspace root")
        if ".." in path.parts:
            raise ValueError("output.run_dir must stay inside the workspace root")
        return path


class OutputSpecV2(OutputSpec):
    emit_baserender_jobs: bool = False
    publish_contract_version: int = 3

    @field_validator("publish_contract_version")
    @classmethod
    def _validate_publish_contract_version(cls, value: int) -> int:
        if int(value) not in {2, 3}:
            raise ValueError("output.publish_contract_version must be 2 or 3")
        return int(value)

    @model_validator(mode="after")
    def _validate_visual_output_dependencies(self) -> "OutputSpecV2":
        if self.emit_baserender_jobs and not self.emit_view_contracts:
            raise ValueError("output.emit_baserender_jobs requires output.emit_view_contracts=true.")
        return self


class PartialComplementRule(StrictBaseModel):
    min_paired_nt: int = Field(ge=1)
    allow_left_tail: bool = True
    allow_right_tail: bool = True


class BulgedCompatibilityRule(StrictBaseModel):
    min_left_paired_nt: int = Field(default=1, ge=1)
    min_right_paired_nt: int = Field(default=1, ge=1)
    max_bulge_nt: int = Field(default=1, ge=0)
    allow_terminal_tails: bool = True


class LigationRuleSpec(StrictBaseModel):
    mode: LigationCompatibilityMode = "exact_complement"
    min_contiguous_core_bp: int = Field(default=1, ge=1)
    max_left_tail_nt: int = Field(default=0, ge=0)
    max_right_tail_nt: int = Field(default=0, ge=0)
    max_bulge_nt: int = Field(default=0, ge=0)
    min_left_flank_bp: int = Field(default=0, ge=0)
    min_right_flank_bp: int = Field(default=0, ge=0)
    bulge_owner: Literal["primary", "complement", "either"] = "either"


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
        supported_invariant_classes: frozenset[str]
        if self.protocol_template == "yiu_circularized_payload_v1":
            supported_invariant_classes = YIU_CANONICAL_TEMPLATE_SUPPORTED_INVARIANT_CLASSES
        else:
            supported_invariant_classes = frozenset()
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
                and invariant.region_ref not in {region.id for region in self.compound_regions}
            ):
                raise ValueError(f"hard_invariants.{invariant.id} references unknown region {invariant.region_ref!r}")
        return self


class YiuSpecDocument(StrictBaseModel):
    yiu: YiuProcessSpec


class YiuSpecDocumentV2(StrictBaseModel):
    yiu: YiuProcessSpecV2


class YiuValidationIssue(StrictBaseModel):
    code: str
    message: str
    step_id: str | None = None
    state_id: str | None = None
    severity: Literal["error", "warning"] = "error"


class ProjectedRegionPart(StrictBaseModel):
    segment_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @field_validator("segment_id")
    @classmethod
    def _validate_segment_id(cls, value: str) -> str:
        return _validate_slug(value, label="projected_region_part.segment_id")

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ProjectedRegionPart":
        if self.end <= self.start:
            raise ValueError("projected_region_part.end must be > projected_region_part.start")
        return self


class ProjectedRegion(StrictBaseModel):
    id: str
    source_region_id: str
    state_id: str
    spans_junction: bool = False
    projection_kind: Literal["atomic", "compound"] = "atomic"
    assembled_coordinate_space: PublishedAssemblySpace | None = None
    parts: list[ProjectedRegionPart] = Field(default_factory=list)

    @field_validator("id", "source_region_id", "state_id")
    @classmethod
    def _validate_id_like_fields(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))

    @model_validator(mode="after")
    def _validate_parts(self) -> "ProjectedRegion":
        if not self.parts:
            raise ValueError("projected_region.parts must be non-empty")
        return self


class YiuPatternEvidenceSummary(StrictBaseModel):
    guaranteed_checks: int = 0
    possible_checks: int = 0
    impossible_checks: int = 0


class YiuStateRecord(StrictBaseModel):
    state_id: str
    step_id: str
    kind: str
    status: Literal["satisfied", "unsatisfied"]
    sequence_mode: SequenceMode = "concrete"
    validation_mode: ValidationMode = "concrete_realization"
    view_contract_version: int | None = None
    state_kind: str | None = None
    topology_kind: TopologyKind | None = None
    primary_sequence: str | None = None
    complement_sequence: str | None = None
    segments: list[dict[str, Any]] = Field(default_factory=list)
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    cuts: list[dict[str, Any]] = Field(default_factory=list)
    junctions: list[dict[str, Any]] = Field(default_factory=list)
    fragments: list[dict[str, Any]] = Field(default_factory=list)
    pattern_evidence_summary: YiuPatternEvidenceSummary = Field(default_factory=YiuPatternEvidenceSummary)
    metadata: dict[str, Any] = Field(default_factory=dict)


class YiuReportMetadata(StrictBaseModel):
    spec_schema_version: int
    step_count: int
    state_count: int
    emitted_view_count: int = 0
    view_contract_version: int | None = None
    catalog_paths: list[str] = Field(default_factory=list)


class YiuValidationReport(StrictBaseModel):
    workflow: Literal["yiu"] = "yiu"
    family: Literal["yiu"] = "yiu"
    protocol: str = "yiu_v1"
    protocol_template: str | None = None
    template_alias_used: str | None = None
    template_alias_status: Literal["deprecated_alias"] | None = None
    workflow_scope: WorkflowScope | None = None
    spec_name: str
    status: Literal["satisfied", "unsatisfied"]
    sequence_mode: SequenceMode = "concrete"
    validation_mode: ValidationMode = "concrete_realization"
    run_dir: str | None = None
    metadata: YiuReportMetadata
    states: list[YiuStateRecord]
    issues: list[YiuValidationIssue] = Field(default_factory=list)


class YiuSolveSourceWindowSpec(StrictBaseModel):
    id: str
    span_ref: str
    alphabet: Literal["dna", "iupac_dna"] = "iupac_dna"
    pattern: str | None = None
    allowed_patterns: list[str] = Field(default_factory=list)

    @field_validator("id", "span_ref")
    @classmethod
    def _validate_id_like(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return normalize_iupac(value)

    @field_validator("allowed_patterns")
    @classmethod
    def _validate_allowed_patterns(cls, value: list[str]) -> list[str]:
        return [normalize_iupac(item) for item in value]

    @model_validator(mode="after")
    def _validate_variable_source(self) -> "YiuSolveSourceWindowSpec":
        if self.pattern is None and not self.allowed_patterns:
            raise ValueError("solve source_window requires pattern or allowed_patterns")
        if self.pattern is not None and self.allowed_patterns:
            raise ValueError("solve source_window must use either pattern or allowed_patterns, not both")
        return self


class YiuSolveVariablesSpec(StrictBaseModel):
    source_windows: list[YiuSolveSourceWindowSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_ids(self) -> "YiuSolveVariablesSpec":
        if not self.source_windows:
            raise ValueError("yiu_solve.variables.source_windows must be non-empty")
        ids = [window.id for window in self.source_windows]
        if len(set(ids)) != len(ids):
            raise ValueError("yiu_solve.variables.source_windows ids must be unique")
        return self


class YiuSolveSearchSpec(StrictBaseModel):
    max_hits: int = Field(default=32, ge=1, le=128)
    materialize_top_k: int = Field(default=8, ge=0, le=128)
    max_search_nodes: int = Field(default=100_000, ge=1, le=1_000_000)
    max_enumerated_candidates: int = Field(default=10_000, ge=1, le=1_000_000)

    @model_validator(mode="after")
    def _validate_limits(self) -> "YiuSolveSearchSpec":
        if self.materialize_top_k > self.max_hits:
            raise ValueError("search.materialize_top_k must be <= search.max_hits")
        return self


class YiuSolveCandidatePolicy(StrictBaseModel):
    require_guaranteed_hard_invariants: bool = True
    forbid_possible_hits: bool = True


class YiuSolveOutputSpec(StrictBaseModel):
    run_dir: Path = Path("outputs/yiu/solve")
    emit_view_contracts: bool = True
    emit_baserender_jobs: bool = True
    publish_contract_version: int = 3

    @field_validator("run_dir")
    @classmethod
    def _validate_run_dir(cls, value: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            raise ValueError("output.run_dir must be relative to the workspace root")
        if ".." in path.parts:
            raise ValueError("output.run_dir must stay inside the workspace root")
        return path

    @field_validator("publish_contract_version")
    @classmethod
    def _validate_publish_contract_version(cls, value: int) -> int:
        if int(value) not in {2, 3}:
            raise ValueError("output.publish_contract_version must be 2 or 3")
        return int(value)

    @model_validator(mode="after")
    def _validate_visual_output_dependencies(self) -> "YiuSolveOutputSpec":
        if self.emit_baserender_jobs and not self.emit_view_contracts:
            raise ValueError("output.emit_baserender_jobs requires output.emit_view_contracts=true.")
        return self


class YiuSolveSpec(StrictBaseModel):
    schema_version: int = 1
    base_spec: Path
    search: YiuSolveSearchSpec = Field(default_factory=YiuSolveSearchSpec)
    variables: YiuSolveVariablesSpec
    candidate_policy: YiuSolveCandidatePolicy = Field(default_factory=YiuSolveCandidatePolicy)
    output: YiuSolveOutputSpec = Field(default_factory=YiuSolveOutputSpec)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if int(value) != 1:
            raise ValueError("yiu_solve.schema_version must be 1")
        return int(value)


class YiuSolveSpecDocument(StrictBaseModel):
    yiu_solve: YiuSolveSpec


class YiuSolveIssue(StrictBaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class YiuSolveHit(StrictBaseModel):
    rank: int = Field(ge=1)
    hit_id: str
    score: list[float | int | str]
    source_sequence: str
    variable_assignments: dict[str, str] = Field(default_factory=dict)
    report_status: Literal["satisfied"] = "satisfied"
    materialized_run_dir: str | None = None
    explicit_design_id: str | None = None
    final_state_id: str | None = None
    final_state_view_path: str | None = None
    final_state_job_path: str | None = None


class YiuSolveReportMetadata(StrictBaseModel):
    max_hits: int = 0
    materialize_top_k: int = 0
    warnings: list[str] = Field(default_factory=list)
    warning_codes: list[str] = Field(default_factory=list)
    search_node_count: int = 0
    enumerated_candidate_count: int = 0
    accepted_candidate_count: int = 0
    returned_hit_count: int = 0
    materialized_hit_count: int = 0
    search_truncated: bool = False
    accepted_pool_truncated: bool = False


class YiuSolveReport(StrictBaseModel):
    workflow: Literal["yiu_solve"] = "yiu_solve"
    family: Literal["yiu"] = "yiu"
    status: Literal["solved", "no_hits", "invalid_spec"]
    solve_id: str | None = None
    spec_path: str
    base_spec_path: str | None = None
    run_dir: str | None = None
    metadata: YiuSolveReportMetadata = Field(default_factory=YiuSolveReportMetadata)
    hits: list[YiuSolveHit] = Field(default_factory=list)
    issues: list[YiuSolveIssue] = Field(default_factory=list)
