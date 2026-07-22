"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/msa/__init__.py

Public MSA API for tool-agnostic alignment bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.aligner.msa.backends.clustalo import (
    ClustalOmegaUnavailableError,
    preflight_clustalo,
    run_clustalo,
)
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


def run_msa(request: MsaRequest) -> MsaRunResult:
    """Run the requested public MSA backend."""

    if request.backend.backend_id == "mafft":
        return run_mafft(request)
    if request.backend.backend_id == "clustalo":
        return run_clustalo(request)
    raise ValueError(f"unsupported MSA backend_id: {request.backend.backend_id!r}")


__all__ = [
    "AlignedFastaBundleManifest",
    "ClustalOmegaUnavailableError",
    "MafftUnavailableError",
    "MsaBackendSpec",
    "MsaRequest",
    "MsaRunResult",
    "MsaVisualizationRequest",
    "MsaVisualizationResult",
    "load_fasta_records",
    "materialize_msa_visualizations",
    "preflight_clustalo",
    "preflight_mafft",
    "run_clustalo",
    "run_mafft",
    "run_msa",
    "validate_aligned_fasta_records",
    "validate_fasta_records",
    "write_bundle_manifest",
    "write_fasta_records",
]
