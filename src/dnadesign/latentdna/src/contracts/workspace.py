"""
Workspace schema contracts for latentdna.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .notebook import NotebookConfig
from .plot import PlotConfig
from .promoter_metadata import PROMOTER_METADATA_DERIVATIONS
from .representations import validate_representation_family_tags, validate_representation_identity

Identifier = Annotated[str, Field(min_length=1)]
NonEmptyText = Annotated[str, Field(min_length=1)]
AggregationMode = Literal["error", "first", "mean", "medoid"]
AlignmentKeyBasis = Literal["record_key", "subject_key"]


class StrictWorkspaceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkspaceSection(StrictWorkspaceModel):
    id: Identifier
    title: str | None = None
    output_root: str


class DefaultsSection(StrictWorkspaceModel):
    analysis_dtype: Literal["float32", "float64"] = "float32"
    metric: str = "cosine"
    random_seed: int = 17
    plot_formats: list[str] = Field(default_factory=lambda: ["svg", "pdf", "png"])
    neighbor_backend: str = "auto"
    memory_policy: "MemoryPolicyConfig" = Field(default_factory=lambda: MemoryPolicyConfig())


class MemoryPolicyConfig(StrictWorkspaceModel):
    warn_fraction_of_system_ram: float = Field(default=0.50, gt=0.0, lt=1.0)
    fail_fraction_of_system_ram: float = Field(default=0.75, gt=0.0, lt=1.0)
    require_override_above_fail: bool = True

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "MemoryPolicyConfig":
        if self.warn_fraction_of_system_ram > self.fail_fraction_of_system_ram:
            raise ValueError("memory_policy warn_fraction_of_system_ram must be <= fail_fraction_of_system_ram")
        return self


class MetadataCopyDerivationConfig(StrictWorkspaceModel):
    kind: Literal["copy"]
    source: str


class MetadataRegexCaptureDerivationConfig(StrictWorkspaceModel):
    kind: Literal["regex_capture"]
    source: str
    pattern: str
    group: int = 1
    default: str | None = None
    normalize: Literal["lower", "upper"] | None = None


class MetadataMapValuesDerivationConfig(StrictWorkspaceModel):
    kind: Literal["map_values"]
    source: str
    mapping: dict[str, str] = Field(min_length=1)
    default: str | None = None


class MetadataCoalesceDerivationConfig(StrictWorkspaceModel):
    kind: Literal["coalesce"]
    sources: list[str] = Field(min_length=1)
    default: str | None = None


class MetadataConstantDerivationConfig(StrictWorkspaceModel):
    kind: Literal["constant"]
    value: str | int | float | bool | None


class MetadataLookupDerivationConfig(StrictWorkspaceModel):
    kind: Literal["lookup"]
    source: str
    left_key: str
    right_key: str
    value_column: str
    missing_policy: Literal["error", "null"] = "error"


MetadataDerivationConfig = Annotated[
    MetadataCopyDerivationConfig
    | MetadataRegexCaptureDerivationConfig
    | MetadataMapValuesDerivationConfig
    | MetadataCoalesceDerivationConfig
    | MetadataConstantDerivationConfig
    | MetadataLookupDerivationConfig,
    Field(discriminator="kind"),
]


class MetadataSection(StrictWorkspaceModel):
    include: list[str] = Field(default_factory=list)
    derivations: dict[str, MetadataDerivationConfig] = Field(default_factory=dict)


class SourceBase(StrictWorkspaceModel):
    kind: str
    record_key: str
    subject_key: str
    context_key: str | None = None
    role: str | None = None
    where: dict[str, Any] | None = None
    metadata_include: list[str] | None = None
    vector_cache_policy: str | None = None
    sequence_scope: str | None = None
    emitted_length_bp: int | None = Field(default=None, gt=0)
    source_interval_length_bp: int | str | None = None
    pooling_span_bp: int | str | None = None
    focal_rule: str | None = None
    window_selection_rule: str | None = None

    @model_validator(mode="after")
    def _validate_sequence_semantics(self) -> "SourceBase":
        for field_name in ("source_interval_length_bp", "pooling_span_bp"):
            value = getattr(self, field_name)
            if isinstance(value, int) and value <= 0:
                raise ValueError(f"{field_name} must be positive when declared as an integer")
        return self


class USRSourceConfig(SourceBase):
    kind: Literal["usr"]
    root: str
    dataset: str


class ParquetSourceConfig(SourceBase):
    kind: Literal["parquet"]
    path: str


class MatrixBundleSourceConfig(SourceBase):
    kind: Literal["matrix_bundle"]
    path: str


class InferFeatureSidecarSourceConfig(SourceBase):
    kind: Literal["infer_feature_sidecar"]
    root: str
    dataset: str


class InferFeatureScalarSidecarSourceConfig(SourceBase):
    kind: Literal["infer_feature_scalar_sidecar"]
    root: str
    dataset: str


SourceConfig = Annotated[
    USRSourceConfig
    | ParquetSourceConfig
    | MatrixBundleSourceConfig
    | InferFeatureSidecarSourceConfig
    | InferFeatureScalarSidecarSourceConfig,
    Field(discriminator="kind"),
]


class VectorColumnSpec(StrictWorkspaceModel):
    kind: Literal["column"]
    name: str


class BundleMatrixSpec(StrictWorkspaceModel):
    kind: Literal["bundle_matrix"]


class VectorDifferenceSpec(StrictWorkspaceModel):
    kind: Literal["vector_difference"]
    left: str
    right: str
    alignment: str


class ConcatenateViewSpec(StrictWorkspaceModel):
    kind: Literal["concatenate"]
    inputs: list[str] = Field(min_length=2)


class AggregateByKeyViewSpec(StrictWorkspaceModel):
    kind: Literal["aggregate_by_key"]
    view: str
    key: str
    aggregation: AggregationMode = "mean"


class ApplyReducerViewSpec(StrictWorkspaceModel):
    kind: Literal["apply_reducer"]
    view: str
    reducer: str


class NormalizeViewSpec(StrictWorkspaceModel):
    kind: Literal["normalize"]
    view: str
    method: Literal["l2", "zscore"] = "l2"


ViewDeriveSpec = Annotated[
    VectorDifferenceSpec | ConcatenateViewSpec | AggregateByKeyViewSpec | ApplyReducerViewSpec | NormalizeViewSpec,
    Field(discriminator="kind"),
]


class SourceBackedViewConfig(StrictWorkspaceModel):
    source: str
    vector: Annotated[VectorColumnSpec | BundleMatrixSpec, Field(discriminator="kind")]
    coordinate_space_id: str
    tags: dict[str, Any] = Field(default_factory=dict)
    role: str | None = None

    @model_validator(mode="after")
    def _validate_representation_family(self) -> "SourceBackedViewConfig":
        validate_representation_identity(self.coordinate_space_id, owner="source-backed view coordinate_space_id")
        validate_representation_family_tags(self.tags, owner="source-backed view")
        return self


class DerivedViewConfig(StrictWorkspaceModel):
    derive: ViewDeriveSpec
    coordinate_space_id: str
    tags: dict[str, Any] = Field(default_factory=dict)
    role: str | None = None

    @model_validator(mode="after")
    def _validate_representation_family(self) -> "DerivedViewConfig":
        validate_representation_identity(self.coordinate_space_id, owner="derived view coordinate_space_id")
        validate_representation_family_tags(self.tags, owner="derived view")
        return self


ViewConfig = SourceBackedViewConfig | DerivedViewConfig


class AlignmentConfig(StrictWorkspaceModel):
    left: str
    right: str
    on: AlignmentKeyBasis | list[str] | None = None
    left_on: list[str] | None = None
    right_on: list[str] | None = None
    support: Literal["intersection"] = "intersection"
    left_aggregation: AggregationMode = "error"
    right_aggregation: AggregationMode = "error"

    @model_validator(mode="after")
    def _validate_key_contract(self) -> "AlignmentConfig":
        has_explicit_pair = self.left_on is not None or self.right_on is not None
        if has_explicit_pair:
            if self.left_on is None or self.right_on is None:
                raise ValueError("alignments must declare both left_on and right_on together")
            if self.on is not None:
                raise ValueError("alignments cannot declare both on and left_on/right_on")
            if not self.left_on or not self.right_on:
                raise ValueError("alignments must declare at least one key column per side")
            if len(self.left_on) != len(self.right_on):
                raise ValueError("alignments left_on and right_on must have the same number of columns")
            return self
        if self.on is None:
            raise ValueError("alignments must declare either on or left_on/right_on")
        if isinstance(self.on, list) and not self.on:
            raise ValueError("alignments with explicit on columns must declare at least one key column")
        return self


class VectorNormScalarSpec(StrictWorkspaceModel):
    kind: Literal["vector_norm"]
    view: str
    norm: Literal["l1", "l2"] = "l2"
    output_column: str | None = None


class ColumnExpressionScalarSpec(StrictWorkspaceModel):
    kind: Literal["column_expression"]
    source: str
    expression: str
    output_column: str


class SelectColumnsScalarSpec(StrictWorkspaceModel):
    kind: Literal["select_columns"]
    source: str
    columns: list[str] = Field(min_length=1)


class RenameColumnsScalarSpec(StrictWorkspaceModel):
    kind: Literal["rename_columns"]
    source: str
    renames: dict[str, str] = Field(min_length=1)


class JoinTablesScalarSpec(StrictWorkspaceModel):
    kind: Literal["join_tables"]
    sources: list[str] = Field(min_length=2)
    on: list[str] = Field(min_length=1)


ScalarDeriveSpec = Annotated[
    VectorNormScalarSpec
    | ColumnExpressionScalarSpec
    | SelectColumnsScalarSpec
    | RenameColumnsScalarSpec
    | JoinTablesScalarSpec,
    Field(discriminator="kind"),
]


class ScalarConfig(StrictWorkspaceModel):
    derive: ScalarDeriveSpec


class LandmarkRepresentationConfig(StrictWorkspaceModel):
    mode: Literal["rows", "centroid", "medoid"]


class LandmarkConfig(StrictWorkspaceModel):
    source: str
    where: dict[str, Any]
    representation: LandmarkRepresentationConfig


class ColumnCohortConfig(StrictWorkspaceModel):
    kind: Literal["column"]
    source: str
    column: str


class PromoterMetadataCohortConfig(StrictWorkspaceModel):
    kind: Literal["promoter_metadata"]
    source: str
    derive: str

    @field_validator("derive")
    @classmethod
    def _validate_derive(cls, value: str) -> str:
        if value not in PROMOTER_METADATA_DERIVATIONS:
            supported = ", ".join(PROMOTER_METADATA_DERIVATIONS)
            raise ValueError(f"unsupported promoter metadata derivation {value!r}; expected one of: {supported}")
        return value


CohortConfig = Annotated[ColumnCohortConfig | PromoterMetadataCohortConfig, Field(discriminator="kind")]


class ReducedViewExportBlockConfig(StrictWorkspaceModel):
    kind: Literal["reduced_view"]
    block_id: str
    source: str
    feature_prefix: str
    alignment: str | None = None
    alignment_aggregation: AggregationMode = "error"
    allowed_use: Literal["benchmark", "eda_only", "both"] = "benchmark"
    leakage_notes: list[str] = Field(default_factory=list)


class TableColumnsExportBlockConfig(StrictWorkspaceModel):
    kind: Literal["table_columns"]
    block_id: str
    source: str
    columns: list[str] = Field(min_length=1)
    feature_prefix: str | None = None
    alignment: str | None = None
    alignment_aggregation: AggregationMode = "error"
    allowed_use: Literal["benchmark", "eda_only", "both"] = "benchmark"
    leakage_notes: list[str] = Field(default_factory=list)


ExportBlockConfig = Annotated[
    ReducedViewExportBlockConfig | TableColumnsExportBlockConfig,
    Field(discriminator="kind"),
]


class ExportConfig(StrictWorkspaceModel):
    row_basis: str
    blocks: list[ExportBlockConfig] = Field(min_length=1)
    matrix_dtype: Literal["float32", "float64"] | None = None
    metadata_columns: list[str] = Field(default_factory=list)


class ReferenceSetSelectorConfig(StrictWorkspaceModel):
    column: str
    equals: str | int | float | bool | None = None
    in_values: list[str | int | float | bool | None] = Field(default_factory=list)
    regex: str | None = None
    not_regex: str | None = None
    non_null: bool = True

    @model_validator(mode="after")
    def _validate_selector(self) -> "ReferenceSetSelectorConfig":
        has_selector = (
            self.equals is not None
            or bool(self.in_values)
            or self.regex is not None
            or self.not_regex is not None
            or self.non_null
        )
        if not has_selector:
            raise ValueError("reference_set selector must declare equals, in_values, regex, not_regex, or non_null")
        return self


class ReferenceSetConfig(StrictWorkspaceModel):
    label: NonEmptyText | None = None
    ids: list[str] = Field(default_factory=list)
    match_column: str = "id"
    label_column: str | None = None
    label_mode: Literal["label_and_highlight", "highlight_only"] = "label_and_highlight"
    display_labels: dict[str, str] = Field(default_factory=dict)
    where: list[ReferenceSetSelectorConfig] = Field(default_factory=list)
    where_all: list[ReferenceSetSelectorConfig] = Field(default_factory=list)
    require_non_empty: bool = True
    notebook_exposed: bool = True

    @model_validator(mode="after")
    def _validate_reference_membership(self) -> "ReferenceSetConfig":
        if not self.ids and not self.where and not self.where_all:
            raise ValueError("reference_sets must declare ids or where selectors")
        return self


class CandidateSetConfig(StrictWorkspaceModel):
    label: NonEmptyText
    description: str | None = None
    views: list[Identifier] = Field(default_factory=list)
    include_tags: dict[str, str] = Field(default_factory=dict)
    exclude_roles: list[str] = Field(default_factory=lambda: ["hidden", "retired"])
    panel_titles: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_membership_rule(self) -> "CandidateSetConfig":
        if not self.views and not self.include_tags:
            raise ValueError("candidate_sets must declare views or include_tags")
        validate_representation_family_tags(self.include_tags, owner="candidate set")
        return self


class AcceptanceCheckConfig(StrictWorkspaceModel):
    kind: Literal[
        "required_plot_kind",
        "required_reference_set",
        "require_reference_set_in_every_panel",
    ]
    value: str | bool


class RecipeStepConfig(StrictWorkspaceModel):
    id: Identifier
    op: str
    depends_on: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class RecipeConfig(StrictWorkspaceModel):
    steps: list[RecipeStepConfig] = Field(min_length=1)


class DeliverableConfig(StrictWorkspaceModel):
    title: NonEmptyText
    section: NonEmptyText
    question: NonEmptyText
    summary: NonEmptyText
    recipe: str
    requires: dict[str, list[str]]
    outputs: dict[str, list[str]]
    docs_refs: list[str]
    acceptance_checks: list[AcceptanceCheckConfig]

    @model_validator(mode="after")
    def _validate_declared_references(self) -> "DeliverableConfig":
        if not self.requires:
            raise ValueError("deliverables must declare at least one requires entry")
        if not self.outputs:
            raise ValueError("deliverables must declare at least one outputs entry")
        return self


class StudyBindingConfig(StrictWorkspaceModel):
    study_id: NonEmptyText
    record_root: NonEmptyText
    deliverable_docs_root: NonEmptyText
    readiness_vocabulary: list[Literal["missing", "attention", "ok"]] = Field(default_factory=list)


class WorkspaceConfig(StrictWorkspaceModel):
    schema_version: Literal["latentdna.workspace.v1"]
    workspace: WorkspaceSection
    defaults: DefaultsSection
    sources: dict[str, SourceConfig]
    metadata: MetadataSection = Field(default_factory=MetadataSection)
    alignments: dict[str, AlignmentConfig] = Field(default_factory=dict)
    views: dict[str, ViewConfig] = Field(default_factory=dict)
    scalars: dict[str, ScalarConfig] = Field(default_factory=dict)
    landmarks: dict[str, LandmarkConfig] = Field(default_factory=dict)
    reference_sets: dict[str, ReferenceSetConfig] = Field(default_factory=dict)
    candidate_sets: dict[str, CandidateSetConfig] = Field(default_factory=dict)
    plots: dict[str, PlotConfig] = Field(default_factory=dict)
    exports: dict[str, ExportConfig] = Field(default_factory=dict)
    cohorts: dict[str, CohortConfig] = Field(default_factory=dict)
    notebooks: dict[str, NotebookConfig] = Field(default_factory=dict)
    deliverables: dict[str, DeliverableConfig] = Field(default_factory=dict)
    recipes: dict[str, RecipeConfig] = Field(default_factory=dict)
    study_binding: StudyBindingConfig | None = None
