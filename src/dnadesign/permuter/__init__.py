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
    MetricSpec,
    NucleotideDmsRequest,
    PermuterResult,
    ProteinDmsRequest,
    ValidationReport,
    VariantRecord,
    default_codon_table_path,
    evaluate_variants,
    generate_variants,
    materialize_result,
    validate_dataset,
)

__version__ = "0.5.0"

__all__ = [
    "CodingDnaDmsRequest",
    "CodingDnaDmsVariantMetadata",
    "DatasetRef",
    "EvaluatorPlan",
    "MetricSpec",
    "NucleotideDmsRequest",
    "PermuterResult",
    "ProteinDmsRequest",
    "ValidationReport",
    "VariantRecord",
    "default_codon_table_path",
    "evaluate_variants",
    "generate_variants",
    "materialize_result",
    "validate_dataset",
    "main",
]


def main() -> int:
    """Run the Permuter CLI entrypoint."""

    from dnadesign.permuter.src.cli.app import main as cli_main

    return cli_main()
