"""
Workspace schema contracts for latentdna.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .notebook import NotebookConfig
from .plot import PlotConfig

Identifier = Annotated[str, Field(min_length=1)]
AggregationMode = Literal["error", "first", "mean", "medoid"]


class StrictWorkspaceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkspaceSection(StrictWorkspaceModel):
    id: Identifier
    output_root: str


class DefaultsSection(StrictWorkspaceModel):
    analysis_dtype: Literal["float32", "float64"] = "float32"
    metric: str = "cosine"
    random_seed: int = 17
    plot_formats: list[str] = Field(default_factory=lambda: ["svg", "png"])
    neighbor_backend: str = "auto"


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
    on: Literal["record_key", "subject_key"] | list[str]
    support: Literal["intersection"] = "intersection"
    left_aggregation: AggregationMode = "error"
    right_aggregation: AggregationMode = "error"


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


class RecipeStepConfig(StrictWorkspaceModel):
    id: Identifier
    op: str
    depends_on: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class RecipeConfig(StrictWorkspaceModel):
    steps: list[RecipeStepConfig] = Field(min_length=1)


class DeliverableConfig(StrictWorkspaceModel):
    kind: str
    description: str
    question: str | None = None
    section: str | None = None
    recipe: str
    requires: dict[str, list[str]] = Field(default_factory=dict)
    outputs: dict[str, list[str]] = Field(default_factory=dict)


class StudyBindingConfig(StrictWorkspaceModel):
    kind: Literal["dnadesign_study"]
    study_dir: str
    readiness_vocabulary: list[Literal["missing", "attention", "ok"]] = Field(
        default_factory=lambda: ["missing", "attention", "ok"]
    )


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
    plots: dict[str, PlotConfig] = Field(default_factory=dict)
    exports: dict[str, ExportConfig] = Field(default_factory=dict)
    cohorts: dict[str, CohortConfig] = Field(default_factory=dict)
    notebooks: dict[str, NotebookConfig] = Field(default_factory=dict)
    deliverables: dict[str, DeliverableConfig] = Field(default_factory=dict)
    recipes: dict[str, RecipeConfig] = Field(default_factory=dict)
    study_binding: StudyBindingConfig | None = None
