"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/tests/msa/test_clustalo_backend.py

Module support for dnadesign.aligner.tests.msa.test_clustalo_backend.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.aligner.msa import (
    ClustalOmegaUnavailableError,
    MsaBackendSpec,
    MsaRequest,
    load_fasta_records,
    run_msa,
)
from dnadesign.aligner.msa.backends.clustalo import preflight_clustalo


def test_clustalo_preflight_fails_without_executable() -> None:
    spec = MsaBackendSpec(backend_id="clustalo", executable="definitely_missing_clustalo_binary")

    with pytest.raises(ClustalOmegaUnavailableError, match="Clustal Omega executable not found"):
        preflight_clustalo(spec)


def test_run_msa_dispatches_clustalo_and_writes_manifest(tmp_path: Path) -> None:
    input_fasta = tmp_path / "input.fasta"
    output_fasta = tmp_path / "aligned.fasta"
    manifest_path = tmp_path / "aligned.manifest.yaml"
    stderr_path = tmp_path / "aligned.stderr.txt"
    input_fasta.write_text(">target\nACDE\n>other\nACDF\n", encoding="utf-8")
    fake_clustalo = _write_fake_clustalo(
        tmp_path / "fake-clustalo",
        """
        if [ "$1" = "--version" ]; then
          echo "1.2.4"
          exit 0
        fi
        output=""
        input=""
        while [ "$#" -gt 0 ]; do
          case "$1" in
            -i) input="$2"; shift 2 ;;
            -o) output="$2"; shift 2 ;;
            *) shift ;;
          esac
        done
        cat "$input" > "$output"
        """,
    )

    result = run_msa(
        MsaRequest(
            input_fasta=input_fasta,
            output_fasta=output_fasta,
            manifest_path=manifest_path,
            target_row_id="target",
            backend=MsaBackendSpec(backend_id="clustalo", executable=str(fake_clustalo)),
            command_args=("--force", "--outfmt=fasta"),
            stderr_path=stderr_path,
            run_label="clade-profile",
        )
    )

    assert result.backend_id == "clustalo"
    assert result.aligned_fasta == output_fasta
    assert load_fasta_records(output_fasta, alphabet="protein", allow_gaps=True)["target"] == "ACDE"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["backend_id"] == "clustalo"
    assert manifest["backend_version"] == "1.2.4"
    assert manifest["target_row_id"] == "target"
    assert manifest["run_label"] == "clade-profile"
    assert manifest["input_fasta_sha256"].startswith("sha256:")
    assert manifest["output_fasta_sha256"].startswith("sha256:")
    assert manifest["stderr_sha256"].startswith("sha256:")


def test_clustalo_request_defaults_to_clustal_omega_args(tmp_path: Path) -> None:
    input_fasta = tmp_path / "input.fasta"
    output_fasta = tmp_path / "aligned.fasta"
    manifest_path = tmp_path / "aligned.manifest.yaml"
    input_fasta.write_text(">target\nACDE\n>other\nACDF\n", encoding="utf-8")
    fake_clustalo = _write_fake_clustalo(
        tmp_path / "fake-clustalo",
        """
        if [ "$1" = "--version" ]; then
          echo "1.2.4"
          exit 0
        fi
        for arg in "$@"; do
          case "$arg" in
            --globalpair|--maxiterate|--reorder) exit 2 ;;
          esac
        done
        output=""
        input=""
        while [ "$#" -gt 0 ]; do
          case "$1" in
            -i) input="$2"; shift 2 ;;
            -o) output="$2"; shift 2 ;;
            *) shift ;;
          esac
        done
        cat "$input" > "$output"
        """,
    )

    run_msa(
        MsaRequest(
            input_fasta=input_fasta,
            output_fasta=output_fasta,
            manifest_path=manifest_path,
            target_row_id="target",
            backend=MsaBackendSpec(backend_id="clustalo", executable=str(fake_clustalo)),
        )
    )

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    command = manifest["command"]
    assert "--globalpair" not in command
    assert "--maxiterate" not in command
    assert "--reorder" not in command
    assert "--outfmt=fasta" in command


def test_run_msa_rejects_clustalo_args_with_io_flags(tmp_path: Path) -> None:
    input_fasta = tmp_path / "input.fasta"
    output_fasta = tmp_path / "aligned.fasta"
    manifest_path = tmp_path / "aligned.manifest.yaml"
    input_fasta.write_text(">target\nACDE\n>other\nACDF\n", encoding="utf-8")
    fake_clustalo = _write_fake_clustalo(
        tmp_path / "fake-clustalo",
        """
        if [ "$1" = "--version" ]; then
          echo "1.2.4"
          exit 0
        fi
        exit 0
        """,
    )

    with pytest.raises(ValueError, match="must not include input/output file flags"):
        run_msa(
            MsaRequest(
                input_fasta=input_fasta,
                output_fasta=output_fasta,
                manifest_path=manifest_path,
                target_row_id="target",
                backend=MsaBackendSpec(backend_id="clustalo", executable=str(fake_clustalo)),
                command_args=("--force", "-i", "bad.fasta"),
            )
        )

    assert not output_fasta.exists()
    assert not manifest_path.exists()


def _write_fake_clustalo(path: Path, body: str) -> Path:
    script = "#!/bin/sh\nset -eu\n" + "\n".join(line.strip() for line in body.strip().splitlines()) + "\n"
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o755)
    return path
