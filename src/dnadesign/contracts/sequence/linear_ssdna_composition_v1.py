"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/sequence/linear_ssdna_composition_v1.py

Generic linear ssDNA composition contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dnadesign.contracts.folding._viennarna_parameters import validate_viennarna_parameters


class SequenceContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _not_blank(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} cannot be empty.")
    return text


class LiteralSourceRefV1(SequenceContractModel):
    kind: Literal["literal"]
    label: str | None = None

    @field_validator("label")
    @classmethod
    def _label_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="source.label")


class DerivedSourceRefV1(SequenceContractModel):
    kind: Literal["derived"]
    from_segment_id: str

    @field_validator("from_segment_id")
    @classmethod
    def _from_segment_id_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="source.from_segment_id")


class UsrSourceRefV1(SequenceContractModel):
    kind: Literal["usr"]
    dataset: str
    root: str | None = None
    record_id: str | None = None
    field: str | None = None

    @field_validator("dataset")
    @classmethod
    def _dataset_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="source.dataset")

    @field_validator("root", "record_id", "field")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="source field")


class RecordSourceRefV1(SequenceContractModel):
    kind: Literal["record"]
    authority: str
    record_id: str

    @field_validator("authority", "record_id")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        return _not_blank(value, label="record source field")


class ArtifactSourceRefV1(SequenceContractModel):
    kind: Literal["artifact"]
    contract: str
    uri: str
    selector: dict[str, Any] = Field(default_factory=dict)
    resolution: dict[str, Any] = Field(default_factory=dict)
    projection: dict[str, Any] = Field(default_factory=dict)

    @field_validator("contract", "uri")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        return _not_blank(value, label="artifact source field")


SourceRefV1 = Annotated[
    LiteralSourceRefV1 | DerivedSourceRefV1 | UsrSourceRefV1 | RecordSourceRefV1 | ArtifactSourceRefV1,
    Field(discriminator="kind"),
]


class ReverseComplementTransformV1(SequenceContractModel):
    kind: Literal["reverse_complement"]
    source_segment_id: str
    assert_expected_sequence: bool = True

    @field_validator("source_segment_id")
    @classmethod
    def _source_segment_id_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="transform.source_segment_id")


SegmentTransformV1 = Annotated[ReverseComplementTransformV1, Field(discriminator="kind")]


class LinearSsdnaSegmentV1(SequenceContractModel):
    segment_id: str
    role: str = "segment"
    sequence: str | None = None
    transform: SegmentTransformV1 | None = None
    source: SourceRefV1 | None = None

    @field_validator("segment_id", "role")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        return _not_blank(value, label="segment field")

    @field_validator("sequence")
    @classmethod
    def _sequence_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="segment.sequence")

    @model_validator(mode="after")
    def _validate_sequence_or_transform(self) -> "LinearSsdnaSegmentV1":
        if self.sequence is None and self.transform is None:
            raise ValueError(f"segment '{self.segment_id}' requires sequence or transform.")
        return self


class LinearSsdnaLocationV1(SequenceContractModel):
    basis: Literal["segment", "unit", "product"]
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    segment_id: str | None = None

    @field_validator("segment_id")
    @classmethod
    def _segment_id_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="location.segment_id")

    @model_validator(mode="after")
    def _validate_location(self) -> "LinearSsdnaLocationV1":
        if self.end <= self.start:
            raise ValueError("location.end must be > location.start.")
        if self.basis == "segment" and self.segment_id is None:
            raise ValueError("location.segment_id is required when basis='segment'.")
        if self.basis != "segment" and self.segment_id is not None:
            raise ValueError("location.segment_id is only allowed when basis='segment'.")
        return self


class LinearSsdnaAnnotationV1(SequenceContractModel):
    annotation_id: str
    role: str = "annotation"
    semantic_label: str | None = None
    location: LinearSsdnaLocationV1

    @field_validator("annotation_id", "role")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        return _not_blank(value, label="annotation field")

    @field_validator("semantic_label")
    @classmethod
    def _semantic_label_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="annotation.semantic_label")


class LinearSsdnaAssertionV1(SequenceContractModel):
    assertion_id: str
    kind: Literal["reverse_complement"]
    left_segment_id: str
    right_segment_id: str
    severity: Literal["error", "warning"] = "error"

    @field_validator("assertion_id", "left_segment_id", "right_segment_id")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        return _not_blank(value, label="assertion field")


class LinearSsdnaUnitV1(SequenceContractModel):
    unit_id: str
    repeat_count: int = Field(default=1, ge=1)
    segments: list[LinearSsdnaSegmentV1] = Field(min_length=1)
    annotations: list[LinearSsdnaAnnotationV1] = Field(default_factory=list)
    assertions: list[LinearSsdnaAssertionV1] = Field(default_factory=list)

    @field_validator("unit_id")
    @classmethod
    def _unit_id_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="unit_id")

    @model_validator(mode="after")
    def _validate_refs(self) -> "LinearSsdnaUnitV1":
        segment_ids: set[str] = set()
        for segment in self.segments:
            if segment.segment_id in segment_ids:
                raise ValueError(f"Duplicate segment_id '{segment.segment_id}'.")
            segment_ids.add(segment.segment_id)
        annotation_ids: set[str] = set()
        for annotation in self.annotations:
            if annotation.annotation_id in annotation_ids:
                raise ValueError(f"Duplicate annotation_id '{annotation.annotation_id}'.")
            annotation_ids.add(annotation.annotation_id)
            if annotation.location.segment_id is not None and annotation.location.segment_id not in segment_ids:
                raise ValueError(
                    f"annotation '{annotation.annotation_id}' references unknown segment_id "
                    f"'{annotation.location.segment_id}'."
                )
        assertion_ids: set[str] = set()
        for assertion in self.assertions:
            if assertion.assertion_id in assertion_ids:
                raise ValueError(f"Duplicate assertion_id '{assertion.assertion_id}'.")
            assertion_ids.add(assertion.assertion_id)
            missing = [
                segment_id
                for segment_id in (assertion.left_segment_id, assertion.right_segment_id)
                if segment_id not in segment_ids
            ]
            if missing:
                joined = ", ".join(missing)
                raise ValueError(f"assertion '{assertion.assertion_id}' references unknown segment_id(s): {joined}.")
        for segment in self.segments:
            if segment.transform is not None and segment.transform.source_segment_id not in segment_ids:
                raise ValueError(
                    f"segment '{segment.segment_id}' transform references unknown segment_id "
                    f"'{segment.transform.source_segment_id}'."
                )
        return self


class LinearSsdnaCanonicalizationV1(SequenceContractModel):
    compare_sequences_case_insensitive: bool = True
    output_sequence_preserves_case: bool = True


class LinearSsdnaQaConfigV1(SequenceContractModel):
    require_no_unknown_bases: bool = True
    allow_degenerate_bases: bool = False
    require_segment_span_coverage: bool = True
    require_non_overlapping_physical_segments: bool = True
    require_annotation_bounds: bool = True
    require_declared_transform_checks: bool = True
    allow_cross_copy_intended_pairings: bool = False


class LinearSsdnaFoldingBackendConfigV1(SequenceContractModel):
    name: Literal["ViennaRNA"]
    interface: Literal["cli", "python_api"] = "cli"
    executable: str | None = None
    python_module: str | None = None
    backend_contract: Literal["secondary_structure_prediction_v2"] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("executable", "python_module")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="folding.backend field")

    @field_validator("parameters", mode="before")
    @classmethod
    def _supported_parameters(cls, value: object) -> dict[str, Any]:
        return validate_viennarna_parameters(value)


class LinearSsdnaFoldingDnaPolicyConfigV1(SequenceContractModel):
    mode: Literal["convert_t_to_u_for_rna_backend", "backend_accepts_dna_directly"]

    @field_validator("mode")
    @classmethod
    def _mode_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="folding.dna_policy.mode")


class LinearSsdnaFoldingConfigV1(SequenceContractModel):
    enabled: bool = False
    required: bool = False
    scope: Literal["canonical_component_unit"] = "canonical_component_unit"
    backend: LinearSsdnaFoldingBackendConfigV1 | None = None
    dna_policy: LinearSsdnaFoldingDnaPolicyConfigV1 | None = None

    @model_validator(mode="after")
    def _validate_enabled_config(self) -> "LinearSsdnaFoldingConfigV1":
        if self.required and not self.enabled:
            raise ValueError("folding.required=true requires folding.enabled=true.")
        if self.enabled and self.backend is None:
            raise ValueError("folding.backend is required when folding.enabled=true.")
        if self.enabled and self.dna_policy is None:
            raise ValueError("folding.dna_policy is required when folding.enabled=true.")
        return self


class LinearSsdnaVisualExportsConfigV1(SequenceContractModel):
    formats: list[str] = Field(default_factory=list)

    @field_validator("formats")
    @classmethod
    def _formats_not_blank(cls, value: list[str]) -> list[str]:
        return [_not_blank(item, label="visual.render_exports.formats item") for item in value]


class LinearSsdnaVisualStyleConfigV1(SequenceContractModel):
    fill: str | None = None
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    edge_color: str | None = None

    @field_validator("fill", "edge_color")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="visual.display_profile style field")


PrimitiveVisualAppliesToV1 = Literal["backbone", "basepair", "nucleotide_text", "section_label", "section_fill"]


class LinearSsdnaPrimitiveVisualRoleV1(SequenceContractModel):
    role_id: str
    display_label: str
    palette_token: str
    stroke_color: str
    fill_color: str
    priority: int = Field(ge=0)
    applies_to: list[PrimitiveVisualAppliesToV1] = Field(default_factory=list)

    @field_validator("role_id", "display_label", "palette_token")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="visual.display_profile.primitive_visual_roles field")

    @field_validator("stroke_color", "fill_color")
    @classmethod
    def _hex_color(cls, value: str) -> str:
        color = _not_blank(value, label="visual.display_profile.primitive_visual_roles color")
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
            raise ValueError("primitive visual role colors must be #RRGGBB hex values.")
        return color

    @field_validator("applies_to")
    @classmethod
    def _applies_to_unique(cls, value: list[PrimitiveVisualAppliesToV1]) -> list[PrimitiveVisualAppliesToV1]:
        if len(set(value)) != len(value):
            raise ValueError("primitive visual role applies_to entries must be unique.")
        return value


class LinearSsdnaDisplayFactV1(SequenceContractModel):
    fact_id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=128)
    value: str = Field(min_length=1, max_length=512)

    @field_validator("fact_id", "label", "value")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="visual.display_profile.fact field")


class LinearSsdnaVisualDisplayProfileV1(SequenceContractModel):
    title: str | None = None
    component_labels: dict[str, str] = Field(default_factory=dict)
    annotation_labels: dict[str, str] = Field(default_factory=dict)
    component_hues: dict[str, str] = Field(default_factory=dict)
    component_styles: dict[str, LinearSsdnaVisualStyleConfigV1] = Field(default_factory=dict)
    primitive_visual_roles: dict[str, LinearSsdnaPrimitiveVisualRoleV1] = Field(default_factory=dict)
    facts: list[LinearSsdnaDisplayFactV1] = Field(default_factory=list, max_length=32)
    overview_hidden_components: list[str] = Field(default_factory=list, max_length=256)
    overview_hidden_annotations: list[str] = Field(default_factory=list, max_length=256)
    base_highlight_color: str | None = None

    @field_validator("title", "base_highlight_color")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="visual.display_profile field")

    @field_validator("component_labels", "annotation_labels", "component_hues")
    @classmethod
    def _mapping_entries_not_blank(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, item in value.items():
            normalized[_not_blank(key, label="visual.display_profile mapping key")] = _not_blank(
                item,
                label="visual.display_profile mapping value",
            )
        return normalized

    @field_validator("primitive_visual_roles")
    @classmethod
    def _primitive_role_keys_match_ids(
        cls, value: dict[str, LinearSsdnaPrimitiveVisualRoleV1]
    ) -> dict[str, LinearSsdnaPrimitiveVisualRoleV1]:
        normalized: dict[str, LinearSsdnaPrimitiveVisualRoleV1] = {}
        for raw_key, role in value.items():
            key = _not_blank(raw_key, label="visual.display_profile.primitive_visual_roles key")
            if key != role.role_id:
                raise ValueError(
                    "visual.display_profile.primitive_visual_roles keys must match each role_id "
                    f"({key!r} != {role.role_id!r})."
                )
            normalized[key] = role
        return normalized

    @field_validator("facts")
    @classmethod
    def _fact_ids_unique(cls, value: list[LinearSsdnaDisplayFactV1]) -> list[LinearSsdnaDisplayFactV1]:
        fact_ids = [fact.fact_id for fact in value]
        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("visual.display_profile.facts fact_id values must be unique.")
        return value

    @field_validator("overview_hidden_components", "overview_hidden_annotations")
    @classmethod
    def _hidden_ids_unique_and_not_blank(cls, value: list[str]) -> list[str]:
        normalized = [_not_blank(item, label="visual.display_profile hidden id") for item in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError("visual.display_profile hidden ids must be unique.")
        return normalized


VisualEmitKindV1 = Literal["sequence_evidence_map_v1", "viennarna_secondary_structure_svg_v1"]
ViennaRNAStructureLayoutV1 = Literal["simple", "naview", "circular", "turtle", "puzzler"]


class LinearSsdnaViennaRNAStructurePlotConfigV1(SequenceContractModel):
    layout_algorithm: ViennaRNAStructureLayoutV1 = "naview"
    python_module: str = "RNA"
    emphasize_stem_base_nucleotides: bool = True

    @field_validator("python_module")
    @classmethod
    def _python_module_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="visual.viennarna_structure_plot.python_module")


class LinearSsdnaVisualConfigV1(SequenceContractModel):
    emit: list[VisualEmitKindV1] = Field(default_factory=lambda: ["sequence_evidence_map_v1"])
    display_profile: LinearSsdnaVisualDisplayProfileV1 = Field(default_factory=LinearSsdnaVisualDisplayProfileV1)
    viennarna_structure_plot: LinearSsdnaViennaRNAStructurePlotConfigV1 = Field(
        default_factory=LinearSsdnaViennaRNAStructurePlotConfigV1
    )
    render_exports: LinearSsdnaVisualExportsConfigV1 = Field(default_factory=LinearSsdnaVisualExportsConfigV1)

    @field_validator("emit", mode="after")
    @classmethod
    def _emit_unique(cls, value: list[VisualEmitKindV1]) -> list[VisualEmitKindV1]:
        if len(set(value)) != len(value):
            raise ValueError("visual.emit entries must be unique.")
        return value


class LinearSsdnaBenchlingExportConfigV1(SequenceContractModel):
    enabled: bool = True
    primary_format: Literal["genbank"] = "genbank"
    sidecars: list[Literal["fasta", "features_csv"]] = Field(default_factory=lambda: ["fasta", "features_csv"])


class LinearSsdnaUsrOutputConfigV1(SequenceContractModel):
    enabled: bool = False


class LinearSsdnaOutputConfigV1(SequenceContractModel):
    workspace: str | None = None
    artifact_bundle: str | None = None
    usr: LinearSsdnaUsrOutputConfigV1 = Field(default_factory=LinearSsdnaUsrOutputConfigV1)

    @field_validator("workspace", "artifact_bundle")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="output path")


class LinearSsdnaCompositionV1(SequenceContractModel):
    contract: Literal["linear_ssdna_composition_v1"] = "linear_ssdna_composition_v1"
    schema_version: Literal[1] = 1
    composition_id: str
    alphabet: Literal["dna"] = "dna"
    topology: Literal["linear_ssdna"] = "linear_ssdna"
    coordinate_system: Literal["zero_based_half_open"] = "zero_based_half_open"
    case_policy: Literal["preserve_input_display_case"] = "preserve_input_display_case"
    canonicalization: LinearSsdnaCanonicalizationV1 = Field(default_factory=LinearSsdnaCanonicalizationV1)
    units: list[LinearSsdnaUnitV1] = Field(min_length=1)
    qa: LinearSsdnaQaConfigV1 = Field(default_factory=LinearSsdnaQaConfigV1)
    folding: LinearSsdnaFoldingConfigV1 = Field(default_factory=LinearSsdnaFoldingConfigV1)
    visual: LinearSsdnaVisualConfigV1 = Field(default_factory=LinearSsdnaVisualConfigV1)
    benchling_export: LinearSsdnaBenchlingExportConfigV1 = Field(default_factory=LinearSsdnaBenchlingExportConfigV1)
    output: LinearSsdnaOutputConfigV1 = Field(default_factory=LinearSsdnaOutputConfigV1)

    @field_validator("composition_id")
    @classmethod
    def _composition_id_not_blank(cls, value: str) -> str:
        return _not_blank(value, label="composition_id")

    @model_validator(mode="after")
    def _validate_units(self) -> "LinearSsdnaCompositionV1":
        unit_ids: set[str] = set()
        for unit in self.units:
            if unit.unit_id in unit_ids:
                raise ValueError(f"Duplicate unit_id '{unit.unit_id}'.")
            unit_ids.add(unit.unit_id)
        if "viennarna_secondary_structure_svg_v1" in self.visual.emit and not self.folding.enabled:
            raise ValueError("visual.emit includes viennarna_secondary_structure_svg_v1 but folding.enabled is false.")
        return self


__all__ = [
    "LinearSsdnaCompositionV1",
    "LinearSsdnaUnitV1",
    "LinearSsdnaSegmentV1",
    "LinearSsdnaAnnotationV1",
    "LinearSsdnaAssertionV1",
    "LinearSsdnaVisualDisplayProfileV1",
    "LinearSsdnaPrimitiveVisualRoleV1",
    "LinearSsdnaDisplayFactV1",
    "LinearSsdnaVisualStyleConfigV1",
]
