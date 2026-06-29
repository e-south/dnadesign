"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/api/__init__.py

Public facade for Permuter generation, evaluation, validation, and handoffs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.permuter.src.api.codon_tables import default_codon_table_path
from dnadesign.permuter.src.api.contracts import (
    CodingDnaDmsRequest,
    CodingDnaDmsVariantMetadata,
    DatasetRef,
    EvaluatorPlan,
    MetricSpec,
    NucleotideDmsRequest,
    PermuterResult,
    ProteinDmsRequest,
    ValidationReport,
    VariantRecord,
)
from dnadesign.permuter.src.api.evaluate import evaluate_variants
from dnadesign.permuter.src.api.generate import generate_variants
from dnadesign.permuter.src.api.infer_handoff import (
    InferFeatureRequest,
    InferFeatureSourceDataset,
    InferSequenceViewSelector,
    infer_feature_request_from_mapping,
    read_infer_feature_request_manifest,
    write_infer_feature_request_manifest,
)
from dnadesign.permuter.src.api.materialize import materialize_result
from dnadesign.permuter.src.api.validate import validate_dataset
from dnadesign.permuter.src.scoring.esmc_masked_marginal import (
    CANONICAL_AMINO_ACIDS,
    MaskedMarginalArtifacts,
    MaskedMarginalJob,
    MaskedMarginalRows,
    build_error_position_row,
    build_masked_marginal_jobs,
    normalize_masked_marginal_response,
    render_masked_marginal_plots,
    validate_masked_marginal_artifacts,
    write_masked_marginal_artifacts,
)

__all__ = [
    "CodingDnaDmsRequest",
    "CodingDnaDmsVariantMetadata",
    "CANONICAL_AMINO_ACIDS",
    "DatasetRef",
    "EvaluatorPlan",
    "InferFeatureRequest",
    "InferFeatureSourceDataset",
    "InferSequenceViewSelector",
    "MetricSpec",
    "MaskedMarginalArtifacts",
    "MaskedMarginalJob",
    "MaskedMarginalRows",
    "NucleotideDmsRequest",
    "PermuterResult",
    "ProteinDmsRequest",
    "ValidationReport",
    "VariantRecord",
    "build_error_position_row",
    "build_masked_marginal_jobs",
    "default_codon_table_path",
    "evaluate_variants",
    "generate_variants",
    "infer_feature_request_from_mapping",
    "materialize_result",
    "normalize_masked_marginal_response",
    "read_infer_feature_request_manifest",
    "render_masked_marginal_plots",
    "validate_dataset",
    "validate_masked_marginal_artifacts",
    "write_masked_marginal_artifacts",
    "write_infer_feature_request_manifest",
]
