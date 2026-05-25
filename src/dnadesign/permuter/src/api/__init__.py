"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/api/__init__.py

Module Author(s): OpenAI Codex
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
]
