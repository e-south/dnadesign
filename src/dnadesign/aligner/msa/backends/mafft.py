"""MAFFT backend wrapper for generic MSA runs."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

from dnadesign.aligner.msa.bundles.manifest import AlignedFastaBundleManifest, write_bundle_manifest
from dnadesign.aligner.msa.contracts import MsaBackendSpec, MsaRequest, MsaRunResult
from dnadesign.aligner.msa.fasta import load_fasta_records
from dnadesign.aligner.msa.validation import validate_aligned_fasta_records


class MafftUnavailableError(RuntimeError):
    """Raised when the declared MAFFT executable is unavailable."""


def preflight_mafft(spec: MsaBackendSpec | None = None) -> tuple[str, str]:
    """Return executable path and version for a declared MAFFT backend."""

    backend = spec or MsaBackendSpec()
    executable_path = shutil.which(backend.executable)
    if executable_path is None:
        raise MafftUnavailableError(f"MAFFT executable not found: {backend.executable}")
    completed = subprocess.run(
        [executable_path, "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    version_text = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0 and not version_text:
        raise MafftUnavailableError(f"MAFFT version check failed for {executable_path}")
    return executable_path, _parse_mafft_version(version_text)


def run_mafft(request: MsaRequest) -> MsaRunResult:
    """Run MAFFT and write an aligned FASTA bundle manifest."""

    executable_path, version = preflight_mafft(request.backend)
    load_fasta_records(request.input_fasta, alphabet="protein", allow_gaps=False)

    request.output_fasta.parent.mkdir(parents=True, exist_ok=True)
    request.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    command = (executable_path, *request.command_args, str(request.input_fasta))
    with request.output_fasta.open("w", encoding="utf-8") as output_handle:
        completed = subprocess.run(
            list(command),
            check=False,
            stdout=output_handle,
            stderr=subprocess.PIPE,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"MAFFT failed with exit code {completed.returncode}: {completed.stderr.strip()}")

    aligned_records = load_fasta_records(request.output_fasta, alphabet="protein", allow_gaps=True)
    validate_aligned_fasta_records(aligned_records, target_row_id=request.target_row_id, alphabet="protein")

    input_hash = _sha256(request.input_fasta)
    output_hash = _sha256(request.output_fasta)
    pixi_lock_hash = _optional_pixi_lock_hash()
    manifest = AlignedFastaBundleManifest(
        backend_id=request.backend.backend_id,
        backend_version=version,
        executable_path=executable_path,
        command=list(command),
        input_fasta=str(request.input_fasta),
        output_fasta=str(request.output_fasta),
        input_fasta_sha256=input_hash,
        output_fasta_sha256=output_hash,
        target_row_id=request.target_row_id,
        environment=request.backend.environment,
        pixi_lock_sha256=pixi_lock_hash,
        failure_policy=request.backend.failure_policy,
    )
    write_bundle_manifest(request.manifest_path, manifest)
    return MsaRunResult(
        aligned_fasta=request.output_fasta,
        manifest_path=request.manifest_path,
        backend_id=request.backend.backend_id,
        backend_version=version,
        command=command,
        input_fasta_sha256=input_hash,
        output_fasta_sha256=output_hash,
        pixi_lock_sha256=pixi_lock_hash,
    )


def _parse_mafft_version(version_text: str) -> str:
    match = re.search(r"v?(\d+(?:\.\d+)+)", version_text)
    if match:
        return match.group(1)
    return version_text.splitlines()[0] if version_text else "unknown"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _optional_pixi_lock_hash() -> str | None:
    for directory in (Path.cwd(), *Path.cwd().parents):
        lock_path = directory / "pixi.lock"
        if lock_path.exists():
            return _sha256(lock_path)
    return None
