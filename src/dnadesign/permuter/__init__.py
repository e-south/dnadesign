"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/__init__.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.permuter.src.api import (
    CodingDnaDmsRequest,
    CodingDnaDmsVariantMetadata,
    DatasetRef,
    EvaluatorPlan,
    InferFeatureRequest,
    InferFeatureSourceDataset,
    InferSequenceViewSelector,
    MetricSpec,
    NucleotideDmsRequest,
    PermuterResult,
    ProteinDmsRequest,
    ValidationReport,
    VariantRecord,
    default_codon_table_path,
    evaluate_variants,
    generate_variants,
    infer_feature_request_from_mapping,
    materialize_result,
    read_infer_feature_request_manifest,
    validate_dataset,
    write_infer_feature_request_manifest,
)

__version__ = "0.5.0"

__all__ = [
    "CodingDnaDmsRequest",
    "CodingDnaDmsVariantMetadata",
    "DatasetRef",
    "EvaluatorPlan",
    "InferFeatureRequest",
    "InferFeatureSourceDataset",
    "InferSequenceViewSelector",
    "MetricSpec",
    "NucleotideDmsRequest",
    "PermuterResult",
    "ProteinDmsRequest",
    "ValidationReport",
    "VariantRecord",
    "default_codon_table_path",
    "evaluate_variants",
    "generate_variants",
    "infer_feature_request_from_mapping",
    "materialize_result",
    "read_infer_feature_request_manifest",
    "validate_dataset",
    "write_infer_feature_request_manifest",
    "main",
]


def main() -> int:
    """Run the Permuter CLI entrypoint."""

    from dnadesign.permuter.src.cli.app import main as cli_main

    return cli_main()
