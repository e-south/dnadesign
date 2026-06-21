"""Public MSA API for tool-agnostic alignment bundles."""

from dnadesign.aligner.msa.backends.mafft import MafftUnavailableError, preflight_mafft, run_mafft
from dnadesign.aligner.msa.bundles.manifest import AlignedFastaBundleManifest, write_bundle_manifest
from dnadesign.aligner.msa.contracts import MsaBackendSpec, MsaRequest, MsaRunResult
from dnadesign.aligner.msa.fasta import load_fasta_records, write_fasta_records
from dnadesign.aligner.msa.validation import validate_aligned_fasta_records, validate_fasta_records
from dnadesign.aligner.msa.visualization import (
    MsaVisualizationRequest,
    MsaVisualizationResult,
    materialize_msa_visualizations,
)

run_msa = run_mafft

__all__ = [
    "AlignedFastaBundleManifest",
    "MafftUnavailableError",
    "MsaBackendSpec",
    "MsaRequest",
    "MsaRunResult",
    "MsaVisualizationRequest",
    "MsaVisualizationResult",
    "load_fasta_records",
    "materialize_msa_visualizations",
    "preflight_mafft",
    "run_mafft",
    "run_msa",
    "validate_aligned_fasta_records",
    "validate_fasta_records",
    "write_bundle_manifest",
    "write_fasta_records",
]
