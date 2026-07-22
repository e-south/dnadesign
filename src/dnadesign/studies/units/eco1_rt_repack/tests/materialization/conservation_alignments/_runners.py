"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/conservation_alignments/_runners.py

MSA runner fixtures for Eco1 conservation-alignment materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from dnadesign.aligner.msa import MsaRequest, MsaRunResult


def recording_copy_runner(observed_requests: list[MsaRequest]):
    def runner(request: MsaRequest) -> MsaRunResult:
        observed_requests.append(request)
        request.output_fasta.parent.mkdir(parents=True, exist_ok=True)
        request.output_fasta.write_text(request.input_fasta.read_text(encoding="utf-8"), encoding="utf-8")
        input_hash = _sha256(request.input_fasta)
        output_hash = _sha256(request.output_fasta)
        request.manifest_path.write_text(
            yaml.safe_dump(
                {
                    "schema_id": "dnadesign.aligner.msa.aligned_fasta_bundle",
                    "schema_version": 1,
                    "backend_id": request.backend.backend_id,
                    "backend_version": "test",
                    "command": [request.backend.backend_id, *request.command_args, str(request.input_fasta)],
                    "input_fasta": str(request.input_fasta),
                    "output_fasta": str(request.output_fasta),
                    "input_fasta_sha256": input_hash,
                    "output_fasta_sha256": output_hash,
                    "target_row_id": request.target_row_id,
                    "environment": request.backend.environment,
                    "pixi_lock_sha256": "sha256:test",
                    "failure_policy": request.backend.failure_policy,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return MsaRunResult(
            aligned_fasta=request.output_fasta,
            manifest_path=request.manifest_path,
            backend_id=request.backend.backend_id,
            backend_version="test",
            command=(request.backend.backend_id, *request.command_args, str(request.input_fasta)),
            input_fasta_sha256=input_hash,
            output_fasta_sha256=output_hash,
            pixi_lock_sha256="sha256:test",
            elapsed_seconds=0.0,
            return_code=0,
            stderr_path=None,
            run_label=request.run_label,
        )

    return runner


def forbidden_runner(request: MsaRequest) -> MsaRunResult:
    raise AssertionError(f"MSA runner should not be called for {request.input_fasta}")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
