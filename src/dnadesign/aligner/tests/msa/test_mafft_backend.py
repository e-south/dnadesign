from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.aligner.msa import MsaBackendSpec, MsaRequest, run_msa
from dnadesign.aligner.msa.backends.mafft import MafftUnavailableError, preflight_mafft


def test_mafft_preflight_fails_without_executable() -> None:
    spec = MsaBackendSpec(backend_id="mafft", executable="definitely_missing_mafft_binary")

    with pytest.raises(MafftUnavailableError, match="MAFFT executable not found"):
        preflight_mafft(spec)


def test_run_msa_writes_aligned_fasta_and_manifest(tmp_path: Path) -> None:
    input_fasta = tmp_path / "input.fasta"
    output_fasta = tmp_path / "aligned.fasta"
    manifest_path = tmp_path / "aligned.manifest.yaml"
    input_fasta.write_text(">target\nACDE\n>other\nACDF\n", encoding="utf-8")

    request = MsaRequest(
        input_fasta=input_fasta,
        output_fasta=output_fasta,
        manifest_path=manifest_path,
        target_row_id="target",
        backend=MsaBackendSpec(backend_id="mafft"),
        command_args=("--auto",),
    )

    try:
        result = run_msa(request)
    except MafftUnavailableError:
        pytest.skip("MAFFT is not installed in the current execution environment")

    assert result.aligned_fasta == output_fasta
    assert output_fasta.exists()
    assert manifest_path.exists()
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "backend_id: mafft" in manifest_text
    assert "target_row_id: target" in manifest_text
    assert "input_fasta_sha256:" in manifest_text
    assert "output_fasta_sha256:" in manifest_text
