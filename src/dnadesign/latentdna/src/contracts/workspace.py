"""
Workspace schema contracts for latentdna.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .notebook import NotebookConfig
from .plot import PlotConfig

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


class MetadataSection(StrictWorkspaceModel):
    include: list[str] = Field(default_factory=list)


class SourceBase(StrictWorkspaceModel):
    kind: str
    record_key: str
    subject_key: str
    context_key: str | None = None
    where: dict[str, Any] | None = None
    metadata_include: list[str] | None = None
    vector_cache_policy: str | None = None


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


SourceConfig = Annotated[USRSourceConfig | ParquetSourceConfig | MatrixBundleSourceConfig, Field(discriminator="kind")]


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


class DerivedViewConfig(StrictWorkspaceModel):
    derive: ViewDeriveSpec
    coordinate_space_id: str
    tags: dict[str, Any] = Field(default_factory=dict)
    role: str | None = None


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
    derive: Literal[
        "design_family",
        "design_regulator_composition",
        "sigma70_variant",
        "campaign_prior",
        "is_control",
        "source_class",
    ]


CohortConfig = Annotated[ColumnCohortConfig | PromoterMetadataCohortConfig, Field(discriminator="kind")]


class ReducedViewExportBlockConfig(StrictWorkspaceModel):
    kind: Literal["reduced_view"]
    block_id: str
    source: str
    feature_prefix: str
    alignment: str | None = None
    alignment_aggregation: AggregationMode = "error"


class TableColumnsExportBlockConfig(StrictWorkspaceModel):
    kind: Literal["table_columns"]
    block_id: str
    source: str
    columns: list[str] = Field(min_length=1)
    feature_prefix: str | None = None
    alignment: str | None = None
    alignment_aggregation: AggregationMode = "error"


ExportBlockConfig = Annotated[
    ReducedViewExportBlockConfig | TableColumnsExportBlockConfig,
    Field(discriminator="kind"),
]


class ExportConfig(StrictWorkspaceModel):
    row_basis: str
    blocks: list[ExportBlockConfig] = Field(min_length=1)
    matrix_dtype: Literal["float32", "float64"] | None = None
    metadata_columns: list[str] = Field(default_factory=list)


class ReferenceSetConfig(StrictWorkspaceModel):
    ids: list[str] = Field(min_length=1)
    match_column: str = "id"
    label_column: str | None = None
    label_mode: Literal["label_and_highlight", "highlight_only"] = "label_and_highlight"


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
    docs_root: NonEmptyText
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
    plots: dict[str, PlotConfig] = Field(default_factory=dict)
    exports: dict[str, ExportConfig] = Field(default_factory=dict)
    cohorts: dict[str, CohortConfig] = Field(default_factory=dict)
    notebooks: dict[str, NotebookConfig] = Field(default_factory=dict)
    deliverables: dict[str, DeliverableConfig] = Field(default_factory=dict)
    recipes: dict[str, RecipeConfig] = Field(default_factory=dict)
    study_binding: StudyBindingConfig | None = None
