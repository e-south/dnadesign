"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/proteinmpnn/__init__.py

ProteinMPNN request adapter primitives for fixed-backbone thread workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.thread.adapters.proteinmpnn.execution import ProteinMpnnExecutionConfig, run_official_proteinmpnn_request
from dnadesign.thread.adapters.proteinmpnn.execution_preflight import (
    proteinmpnn_git_commit,
    resolve_proteinmpnn_root,
    validate_proteinmpnn_root,
)
from dnadesign.thread.adapters.proteinmpnn.manifest import (
    build_request_manifest,
    proteinmpnn_run_commands,
    request_hash,
)
from dnadesign.thread.adapters.proteinmpnn.models import (
    ProteinMpnnBackboneExport,
    ProteinMpnnRequestIssue,
    ProteinMpnnRunArtifacts,
)
from dnadesign.thread.adapters.proteinmpnn.positions import (
    mapped_chain_rows,
    require_int_list,
    require_missing_backbone_excluded,
    to_proteinmpnn_positions,
)
from dnadesign.thread.adapters.proteinmpnn.samples import (
    parse_proteinmpnn_fasta_samples,
    parse_proteinmpnn_outputs,
    validate_sample_table,
    write_backend_run_manifest,
    write_sample_table,
)
from dnadesign.thread.adapters.proteinmpnn.sidecars import (
    assigned_chains_payload,
    fixed_positions_payload,
    resolve_manifest_sidecar_path,
    resolve_manifest_sidecar_paths,
    write_jsonl,
)
from dnadesign.thread.adapters.proteinmpnn.structure import export_chain_backbone
from dnadesign.thread.adapters.proteinmpnn.validation import validate_request_manifest

__all__ = [
    "ProteinMpnnBackboneExport",
    "ProteinMpnnExecutionConfig",
    "ProteinMpnnRequestIssue",
    "ProteinMpnnRunArtifacts",
    "assigned_chains_payload",
    "build_request_manifest",
    "export_chain_backbone",
    "fixed_positions_payload",
    "mapped_chain_rows",
    "parse_proteinmpnn_fasta_samples",
    "parse_proteinmpnn_outputs",
    "proteinmpnn_git_commit",
    "proteinmpnn_run_commands",
    "request_hash",
    "require_int_list",
    "require_missing_backbone_excluded",
    "resolve_manifest_sidecar_path",
    "resolve_manifest_sidecar_paths",
    "resolve_proteinmpnn_root",
    "run_official_proteinmpnn_request",
    "to_proteinmpnn_positions",
    "validate_request_manifest",
    "validate_proteinmpnn_root",
    "validate_sample_table",
    "write_backend_run_manifest",
    "write_jsonl",
    "write_sample_table",
]
