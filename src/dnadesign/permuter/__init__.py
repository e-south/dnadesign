"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/__init__.py

Package exports for Permuter.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.permuter.src.api import (
    CANONICAL_AMINO_ACIDS,
    CodingDnaDmsRequest,
    CodingDnaDmsVariantMetadata,
    DatasetRef,
    EvaluatorPlan,
    InferFeatureRequest,
    InferFeatureSourceDataset,
    InferSequenceViewSelector,
    MaskedMarginalArtifacts,
    MaskedMarginalJob,
    MaskedMarginalRows,
    MetricSpec,
    NucleotideDmsRequest,
    PermuterResult,
    ProteinDmsRequest,
    ValidationReport,
    VariantRecord,
    build_error_position_row,
    build_masked_marginal_jobs,
    default_codon_table_path,
    evaluate_variants,
    generate_variants,
    infer_feature_request_from_mapping,
    materialize_result,
    normalize_masked_marginal_response,
    read_infer_feature_request_manifest,
    render_masked_marginal_plots,
    validate_dataset,
    validate_masked_marginal_artifacts,
    write_infer_feature_request_manifest,
    write_masked_marginal_artifacts,
)

__version__ = "0.5.0"

__all__ = [
    "CodingDnaDmsRequest",
    "CodingDnaDmsVariantMetadata",
    "CANONICAL_AMINO_ACIDS",
    "DatasetRef",
    "EvaluatorPlan",
    "InferFeatureRequest",
    "InferFeatureSourceDataset",
    "InferSequenceViewSelector",
    "MaskedMarginalArtifacts",
    "MaskedMarginalJob",
    "MaskedMarginalRows",
    "MetricSpec",
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
    "main",
]


def main() -> int:
    """Run the Permuter CLI entrypoint."""

    from dnadesign.permuter.src.cli.app import main as cli_main

    return cli_main()
