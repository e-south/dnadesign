"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/models.py

Strict schema and report contracts for the YIU workflow family.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.bio.iupac import normalize_iupac
from dnadesign.cruncher.config.schema_v3 import StrictBaseModel

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
            "adapter_ligation": ("adapter_sequence",),
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


class YiuSpecDocument(StrictBaseModel):
    yiu: YiuProcessSpec


class YiuValidationIssue(StrictBaseModel):
    code: str
    message: str
    step_id: str | None = None
    state_id: str | None = None
    severity: Literal["error", "warning"] = "error"


class YiuStateRecord(StrictBaseModel):
    state_id: str
    step_id: str
    kind: str
    status: Literal["satisfied", "unsatisfied"]
    primary_sequence: str | None = None
    complement_sequence: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class YiuReportMetadata(StrictBaseModel):
    spec_schema_version: int
    step_count: int
    state_count: int
    emitted_view_count: int = 0
    catalog_paths: list[str] = Field(default_factory=list)


class YiuValidationReport(StrictBaseModel):
    workflow: Literal["yiu"] = "yiu"
    protocol: Literal["yiu_v1"] = "yiu_v1"
    spec_name: str
    status: Literal["satisfied", "unsatisfied"]
    run_dir: str | None = None
    metadata: YiuReportMetadata
    states: list[YiuStateRecord]
    issues: list[YiuValidationIssue] = Field(default_factory=list)
